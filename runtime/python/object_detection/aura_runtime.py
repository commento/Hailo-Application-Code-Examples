from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from subprocess import TimeoutExpired
import os
import platform
import queue
import subprocess
import time
import wave

import cv2
import numpy as np
from loguru import logger

from object_detection_post_process import extract_detections

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional on machines without audio stack
    sd = None


@dataclass
class AudioFeatures:
    rms: float = 0.0
    peak: float = 0.0


@dataclass
class TrackedPerformer:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    age: int


class AudioAnalyzer:
    def __init__(self, sample_rate: int, block_size: int, device: str | int | None = None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._features = AudioFeatures()
        self._lock = Lock()
        self._stream = None
        self._wave_file: wave.Wave_write | None = None
        self._record_queue: Queue[bytes] | None = None
        self._record_thread: Thread | None = None
        self._recorded_frames = 0
        self._stream_started = False

    def start(self) -> None:
        if sd is None:
            logger.warning("sounddevice is not available; aura audio gate will stay silent.")
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._stream_started = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._record_thread is not None and self._record_queue is not None:
            self._record_queue.put(b"")
            self._record_thread.join(timeout=2)
            self._record_thread = None
            self._record_queue = None
        if self._wave_file is not None:
            self._wave_file.close()
            self._wave_file = None

    def start_recording(self, path: str) -> None:
        if self._wave_file is not None:
            self._wave_file.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._wave_file = wave.open(path, "wb")
        self._wave_file.setnchannels(1)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
        self._recorded_frames = 0
        self._record_queue = Queue(maxsize=64)
        self._record_thread = Thread(target=self._record_worker, daemon=True)
        self._record_thread.start()

    def read(self) -> AudioFeatures:
        with self._lock:
            return AudioFeatures(**self._features.__dict__)

    @property
    def stream_started(self) -> bool:
        return self._stream_started

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - realtime callback
        samples = np.squeeze(indata.astype(np.float32))
        if samples.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        with self._lock:
            self._features = AudioFeatures(rms=rms, peak=peak)

        if self._wave_file is None or self._record_queue is None:
            return

        pcm16 = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        try:
            self._record_queue.put_nowait(pcm16.tobytes())
            self._recorded_frames += int(samples.size)
        except Exception:
            pass

    def _record_worker(self) -> None:
        if self._record_queue is None:
            return
        while True:
            try:
                chunk = self._record_queue.get(timeout=0.2)
            except Empty:
                continue
            if chunk == b"":
                break
            if self._wave_file is not None:
                self._wave_file.writeframes(chunk)


class FfmpegRecorder:
    def __init__(
        self,
        ffmpeg_bin: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        video_codec: str = "libx264",
        pixel_format: str = "yuv420p",
        crf: int = 20,
        preset: str = "veryfast",
    ):
        self.output_file = Path(datetime.now().strftime(output_path))
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.video_only_file = self.output_file.with_suffix(".video.mp4")
        self.width = width
        self.height = height
        self.fps = max(1.0, float(fps))
        self._frame_interval = 1.0 / self.fps
        self._latest_frame: np.ndarray | None = None
        self._lock = Lock()
        self._stop_event = Event()
        self.process = self._start_process(ffmpeg_bin, video_codec, pixel_format, crf, preset)
        self.ffmpeg_bin = ffmpeg_bin
        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _start_process(
        self,
        ffmpeg_bin: str,
        video_codec: str,
        pixel_format: str,
        crf: int,
        preset: str,
    ) -> subprocess.Popen:
        command = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.3f}",
            "-i",
            "-",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format,
            "-an",
            str(self.video_only_file),
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame_bgr.copy()

    def close(self) -> None:
        self._stop_event.set()
        self._writer_thread.join(timeout=2)
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def finalize(self, audio_path: str | None) -> None:
        if audio_path:
            self._mux_audio(audio_path)
        else:
            self._promote_video_only()

    def _writer_loop(self) -> None:
        next_frame_time = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is not None and self.process.stdin is not None:
                try:
                    self.process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    self._stop_event.set()
                    break

            next_frame_time += self._frame_interval
            sleep_for = next_frame_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_time = time.monotonic()

    def _promote_video_only(self) -> None:
        if not self.video_only_file.exists() or self.video_only_file.stat().st_size == 0:
            return
        if self.output_file.exists():
            self.output_file.unlink()
        self.video_only_file.replace(self.output_file)

    def _mux_audio(self, audio_path: str) -> None:
        audio_file = Path(audio_path)
        if not self.video_only_file.exists() or self.video_only_file.stat().st_size == 0:
            return
        if not audio_file.exists() or audio_file.stat().st_size == 0:
            self._promote_video_only()
            return

        muxed_tmp = self.output_file.with_suffix(".mux.mp4")
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(self.video_only_file),
            "-i",
            str(audio_file),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(muxed_tmp),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            logger.warning("Audio mux failed; keeping silent video.")
            self._promote_video_only()
            return
        muxed_tmp.replace(self.output_file)
        if self.video_only_file.exists():
            self.video_only_file.unlink()


class RisingAuraRenderer:
    def __init__(
        self,
        aura_radius: int,
        aura_alpha: float,
        background_dim: float,
        audio_threshold: float,
        audio_scale: float,
        render_scale: float = 0.5,
        person_edges: bool = False,
        edge_warp: bool = False,
        edge_warp_strength: float = 0.34,
        edge_warp_scale: float = 0.5,
    ):
        self.aura_radius = aura_radius
        self.aura_alpha = aura_alpha
        self.background_dim = background_dim
        self.audio_threshold = audio_threshold
        self.audio_scale = audio_scale
        self.render_scale = render_scale
        self.person_edges = person_edges
        self.edge_warp = edge_warp
        self.edge_warp_strength = edge_warp_strength
        self.edge_warp_scale = edge_warp_scale
        self.aura_states: dict[int, float] = {}
        self.age_states: dict[int, int] = {}
        self._plume_layer: np.ndarray | None = None
        self._smoothed_audio_gate = 0.0
        self._scene_energy = 0.0

    def render(self, frame_rgb: np.ndarray, performers: list[TrackedPerformer], audio: AudioFeatures) -> np.ndarray:
        aura_level = self._audio_gate(audio)
        if aura_level <= 0.001 or not performers:
            self.aura_states.clear()
            self._scene_energy = self._ease(self._scene_energy, 0.0, attack=0.0, release=0.04)
            return frame_rgb.copy()

        work_frame, work_performers, scale = self._prepare_aura_workspace(frame_rgb, performers)
        self._ensure_plume(work_frame)
        mist = np.zeros_like(work_frame)
        active_ids = set()

        for performer in work_performers:
            active_ids.add(performer.track_id)
            previous = self.aura_states.get(performer.track_id, 0.0)
            age_boost = min(1.0, performer.age / 8.0)
            target = aura_level * (0.35 + 0.65 * age_boost)
            presence = self._ease(previous, target, attack=0.055, release=0.045)
            self.aura_states[performer.track_id] = presence
            if presence <= 0.02:
                continue
            self._draw_person_aura(mist, frame_rgb, performer, presence, audio)

        for track_id in list(self.aura_states.keys()):
            if track_id not in active_ids:
                faded = self._ease(self.aura_states[track_id], 0.0, attack=0.0, release=0.045)
                if faded <= 0.02:
                    del self.aura_states[track_id]
                else:
                    self.aura_states[track_id] = faded

        active_presence = max(self.aura_states.values(), default=0.0)
        self._scene_energy = self._ease(self._scene_energy, active_presence, attack=0.035, release=0.035)

        plume = self._update_plume_layer(mist)
        cv2.add(mist, plume, dst=mist)
        mist = self._soft_blur(mist, sigma=max(6.0, 18.0 * scale))
        if scale < 0.999:
            mist = cv2.resize(mist, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        base = frame_rgb.copy()
        if self.background_dim > 0.0:
            base = cv2.convertScaleAbs(base, alpha=max(0.0, 1.0 - self.background_dim), beta=0)
        if self.edge_warp and self.edge_warp_strength > 0.0:
            base = self._apply_edge_fisheye(base, self.edge_warp_strength * self._scene_energy)
        return cv2.addWeighted(base, 1.0, mist, self.aura_alpha, 0.0)

    def _audio_gate(self, audio: AudioFeatures) -> float:
        rms_energy = max(0.0, (audio.rms - self.audio_threshold) * self.audio_scale)
        peak_energy = max(0.0, (audio.peak - self.audio_threshold * 1.1) * (self.audio_scale * 0.12))
        target = min(1.0, rms_energy * 0.94 + peak_energy * 0.06)
        self._smoothed_audio_gate = self._ease(
            self._smoothed_audio_gate,
            target,
            attack=0.035,
            release=0.028,
        )
        return self._smoothed_audio_gate

    def _draw_person_aura(
        self,
        mist: np.ndarray,
        frame_rgb: np.ndarray,
        performer: TrackedPerformer,
        presence: float,
        audio: AudioFeatures,
    ) -> None:
        x, y, w, h = performer.bbox
        cx, _ = performer.center
        radius = int(self.aura_radius * (0.75 + presence * 0.7 + min(audio.peak, 0.2)))
        color = self._aura_tone(presence)

        if self.person_edges:
            mask, x0, y0 = self._person_edge_mask(frame_rgb, performer)
            if mask is not None:
                upper = np.zeros_like(mask)
                cv2.rectangle(upper, (0, 0), (upper.shape[1], max(1, int(upper.shape[0] * 0.58))), 255, -1)
                mask = cv2.bitwise_and(mask, upper)
                halo = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
                halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=3.5, sigmaY=3.5)
                self._apply_mask(mist, halo, x0, y0, color)

        shoulder_y = int(y + h * 0.28)
        shoulder_axes = (max(18, int(w * 0.46)), max(12, int(h * (0.10 + presence * 0.05))))
        cv2.ellipse(mist, (int(cx), shoulder_y), shoulder_axes, 0, 0, 360, color, -1, cv2.LINE_AA)

        head_y = max(0, int(y + h * 0.16))
        head_axes = (max(12, int(radius * 0.16)), max(18, int(radius * 0.22)))
        cv2.ellipse(mist, (int(cx), head_y), head_axes, 0, 0, 360, color, -1, cv2.LINE_AA)

        beam_top = max(0, int(y - radius * (1.2 + presence * 0.8)))
        beam_bottom = max(0, int(y + h * 0.22))
        beam_width = max(12, int(w * (0.12 + presence * 0.10)))
        points = np.array(
            [
                [int(cx - beam_width * 0.3), beam_top],
                [int(cx + beam_width * 0.3), beam_top],
                [int(cx + beam_width), beam_bottom],
                [int(cx - beam_width), beam_bottom],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mist, points, tuple(int(channel * 0.62) for channel in color), cv2.LINE_AA)

    def _person_edge_mask(
        self,
        frame_rgb: np.ndarray,
        performer: TrackedPerformer,
    ) -> tuple[np.ndarray | None, int, int]:
        x, y, w, h = performer.bbox
        pad_x = max(8, int(w * 0.12))
        pad_y = max(8, int(h * 0.08))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame_rgb.shape[1], x + w + pad_x)
        y1 = min(frame_rgb.shape[0], y + h + pad_y)
        roi = frame_rgb[y0:y1, x0:x1]
        if roi.size == 0:
            return None, x0, y0

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 110)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        if cv2.countNonZero(edges) == 0:
            return None, x0, y0
        return edges, x0, y0

    def _update_plume_layer(self, mist: np.ndarray) -> np.ndarray:
        assert self._plume_layer is not None
        shifted = np.zeros_like(self._plume_layer)
        rise_px = 8
        if rise_px < shifted.shape[0]:
            shifted[:-rise_px] = self._plume_layer[rise_px:]

        spread = cv2.GaussianBlur(shifted, (0, 0), sigmaX=5.5, sigmaY=7.0)
        spread = cv2.convertScaleAbs(spread, alpha=0.82, beta=0)

        h, w = mist.shape[:2]
        upper_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(upper_mask, (0, 0), (w, max(1, int(h * 0.48))), 255, -1)
        seed = cv2.bitwise_and(mist, mist, mask=upper_mask)
        seed = cv2.GaussianBlur(seed, (0, 0), sigmaX=4.0, sigmaY=4.0)
        seed = cv2.convertScaleAbs(seed, alpha=0.26, beta=0)
        cv2.add(spread, seed, dst=spread)
        self._plume_layer = spread
        return spread

    def _ensure_plume(self, frame: np.ndarray) -> None:
        if self._plume_layer is None or self._plume_layer.shape != frame.shape:
            self._plume_layer = np.zeros_like(frame)

    def _prepare_aura_workspace(
        self,
        frame_rgb: np.ndarray,
        performers: list[TrackedPerformer],
    ) -> tuple[np.ndarray, list[TrackedPerformer], float]:
        scale = float(np.clip(self.render_scale, 0.25, 1.0))
        if scale >= 0.999:
            return frame_rgb, performers, 1.0

        h, w = frame_rgb.shape[:2]
        work_w = max(16, int(w * scale))
        work_h = max(16, int(h * scale))
        work_frame = cv2.resize(frame_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
        scaled = []
        for performer in performers:
            x, y, bw, bh = performer.bbox
            scaled.append(
                TrackedPerformer(
                    track_id=performer.track_id,
                    bbox=(
                        int(x * scale),
                        int(y * scale),
                        max(1, int(bw * scale)),
                        max(1, int(bh * scale)),
                    ),
                    center=(int(performer.center[0] * scale), int(performer.center[1] * scale)),
                    age=performer.age,
                )
            )
        return work_frame, scaled, scale

    def _aura_tone(self, presence: float) -> tuple[int, int, int]:
        return (
            min(235, 135 + int(presence * 45)),
            min(245, 148 + int(presence * 65)),
            min(255, 180 + int(presence * 70)),
        )

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray, x0: int, y0: int, color: tuple[int, int, int]) -> None:
        roi = image[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        colored = np.zeros_like(roi)
        colored[:] = color
        masked = cv2.bitwise_and(colored, colored, mask=mask)
        cv2.add(roi, masked, dst=roi)

    def _soft_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, sigma / 2.0), sigmaY=max(1.0, sigma / 2.0))
        return cv2.resize(blurred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _apply_edge_fisheye(self, frame: np.ndarray, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.002:
            return frame

        full_h, full_w = frame.shape[:2]
        warp_scale = float(np.clip(self.edge_warp_scale, 0.2, 1.0))
        if warp_scale < 0.999:
            work_w = max(16, int(full_w * warp_scale))
            work_h = max(16, int(full_h * warp_scale))
            work_frame = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work_w, work_h = full_w, full_h
            work_frame = frame

        grid_scale = 0.25
        grid_w = max(8, int(work_w * grid_scale))
        grid_h = max(8, int(work_h * grid_scale))
        yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        xx *= work_w / grid_w
        yy *= work_h / grid_h

        cx = work_w * 0.5
        cy = work_h * 0.5
        norm_x = (xx - cx) / max(work_w * 0.5, 1.0)
        norm_y = (yy - cy) / max(work_h * 0.5, 1.0)
        radius = np.sqrt(norm_x * norm_x + norm_y * norm_y)

        edge = np.clip((radius - 0.58) / 0.42, 0.0, 1.0)
        edge = edge * edge * (3.0 - 2.0 * edge)
        stretch = 1.0 + edge * strength * 0.34

        map_x_small = cx + norm_x * stretch * (work_w * 0.5)
        map_y_small = cy + norm_y * stretch * (work_h * 0.5)
        map_x = cv2.resize(map_x_small, (work_w, work_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_small, (work_w, work_h), interpolation=cv2.INTER_CUBIC)
        map_x = np.clip(map_x, 0, work_w - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, work_h - 1).astype(np.float32)
        warped = cv2.remap(work_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        if warp_scale < 0.999:
            warped = cv2.resize(warped, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        return warped

    def _ease(self, current: float, target: float, attack: float, release: float) -> float:
        if target > current:
            return current + (target - current) * attack
        return current + (target - current) * release


class AuraPostProcessor:
    def __init__(self, labels, config_data, tracker, audio: AudioAnalyzer, renderer: RisingAuraRenderer, draw_boxes: bool):
        self.labels = labels
        self.config_data = config_data
        self.tracker = tracker
        self.audio = audio
        self.renderer = renderer
        self.draw_boxes = draw_boxes
        self._frame_age = 0

    def __call__(self, original_frame, infer_results):
        detections = extract_detections(original_frame, infer_results, self.config_data)
        performers = self._performers_from_detections(detections)
        output = self.renderer.render(original_frame, performers, self.audio.read())
        if self.draw_boxes:
            self._draw_debug_boxes(output, performers)
        return output

    def _performers_from_detections(self, detections: dict) -> list[TrackedPerformer]:
        boxes = detections["detection_boxes"]
        scores = detections["detection_scores"]
        classes = detections["detection_classes"]
        person_indices = [
            idx for idx, class_id in enumerate(classes)
            if class_id < len(self.labels) and self.labels[class_id].lower() == "person"
        ]

        if self.tracker is not None:
            dets_for_tracker = np.array([[*boxes[idx], scores[idx]] for idx in person_indices], dtype=np.float32)
            if dets_for_tracker.size == 0:
                return []
            tracks = self.tracker.update(dets_for_tracker)
            performers = []
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.tlbr)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                performers.append(
                    TrackedPerformer(
                        track_id=int(track.track_id),
                        bbox=(x1, y1, w, h),
                        center=(x1 + w // 2, y1 + h // 2),
                        age=int(getattr(track, "tracklet_len", 1) or 1),
                    )
                )
            return performers

        self._frame_age += 1
        performers = []
        for idx in person_indices:
            x1, y1, x2, y2 = map(int, boxes[idx])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            performers.append(
                TrackedPerformer(
                    track_id=idx + 1,
                    bbox=(x1, y1, w, h),
                    center=(x1 + w // 2, y1 + h // 2),
                    age=self._frame_age,
                )
            )
        return performers

    def _draw_debug_boxes(self, image: np.ndarray, performers: list[TrackedPerformer]) -> None:
        for performer in performers:
            x, y, w, h = performer.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (180, 240, 255), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"person {performer.track_id}",
                (x + 4, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def visualize_with_aura_recording(
    output_queue: queue.Queue,
    cap,
    save_stream_output: bool,
    output_dir: str,
    callback,
    audio: AudioAnalyzer,
    fps_tracker=None,
    output_resolution=None,
    framerate: float | None = None,
    record_audio_video: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    recording_output: str | None = None,
    video_codec: str = "libx264",
    pixel_format: str = "yuv420p",
    crf: int = 20,
    preset: str = "veryfast",
    stop_event=None,
) -> None:
    image_id = 0
    recorder = None
    audio_path = None
    frame_width = None
    frame_height = None

    try:
        show_window = cap is not None
        if show_window:
            try:
                cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_FREERATIO"):
                    cv2.setWindowProperty("Output", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
                cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow("Output", 0, 0)
            except cv2.error:
                logger.warning("Could not create the OpenCV preview window. Continuing without live preview.")
                show_window = False

            base_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            base_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            if output_resolution is not None:
                frame_width, frame_height = output_resolution
            else:
                frame_width, frame_height = base_width, base_height
            display_size = _detect_display_size() or (frame_width, frame_height)

            if record_audio_video:
                cam_fps = cap.get(cv2.CAP_PROP_FPS)
                final_fps = framerate or (cam_fps if cam_fps and cam_fps > 1 else 30.0)
                os.makedirs(output_dir, exist_ok=True)
                recording_output = recording_output or os.path.join(output_dir, "aura_%Y%m%d_%H%M%S.mp4")
                recorder = FfmpegRecorder(
                    ffmpeg_bin=ffmpeg_bin,
                    output_path=recording_output,
                    width=frame_width,
                    height=frame_height,
                    fps=final_fps,
                    video_codec=video_codec,
                    pixel_format=pixel_format,
                    crf=crf,
                    preset=preset,
                )
                audio_path = str(recorder.output_file.with_suffix(".wav"))
                audio.start_recording(audio_path)

        while stop_event is None or not stop_event.is_set():
            try:
                result = output_queue.get(timeout=0.1)
            except Empty:
                continue
            while True:
                try:
                    newer_result = output_queue.get_nowait()
                except Empty:
                    break
                output_queue.task_done()
                result = newer_result

            if result is None:
                output_queue.task_done()
                break

            original_frame, inference_result, *rest = result
            if isinstance(inference_result, list) and len(inference_result) == 1:
                inference_result = inference_result[0]

            if rest:
                frame_with_detections = callback(original_frame, inference_result, rest[0])
            else:
                frame_with_detections = callback(original_frame, inference_result)

            if fps_tracker is not None:
                fps_tracker.increment()

            bgr_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
            frame_to_show = _resize_frame_for_output(bgr_frame, output_resolution)

            if cap is not None:
                if show_window:
                    try:
                        cv2.imshow("Output", _prepare_fullscreen_frame(frame_to_show, display_size))
                    except cv2.error:
                        logger.warning("OpenCV preview failed. Disabling live preview.")
                        show_window = False
                if recorder is not None and frame_width and frame_height:
                    recorder.write(cv2.resize(frame_to_show, (frame_width, frame_height)))
            else:
                cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_to_show)

            image_id += 1
            output_queue.task_done()
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    if stop_event is not None:
                        stop_event.set()
                    break
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if recorder is not None:
            recorder.close()
            audio.stop()
            recorder.finalize(audio_path)
            logger.success(f"Aura performance saved to {recorder.output_file}")


def _resize_frame_for_output(frame: np.ndarray, resolution):
    if resolution is None:
        return frame

    target_w, target_h = resolution
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _prepare_fullscreen_frame(frame: np.ndarray, display_size: tuple[int, int] | None) -> np.ndarray:
    if display_size is None:
        return frame

    display_w, display_h = display_size
    frame_h, frame_w = frame.shape[:2]
    if display_w <= 0 or display_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return frame

    scale = max(display_w / frame_w, display_h / frame_h)
    out_w = max(1, int(frame_w * scale))
    out_h = max(1, int(frame_h * scale))
    resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (out_w - display_w) // 2)
    y0 = max(0, (out_h - display_h) // 2)
    return resized[y0:y0 + display_h, x0:x0 + display_w]


def _detect_display_size() -> tuple[int, int] | None:
    env_size = os.environ.get("AURA_DISPLAY_SIZE")
    if env_size and "x" in env_size:
        width_text, height_text = env_size.lower().split("x", 1)
        try:
            return int(width_text), int(height_text)
        except ValueError:
            pass

    if platform.system() != "Linux":
        return None

    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    for line in result.stdout.splitlines():
        if "*" not in line:
            continue
        parts = line.strip().split()
        if not parts or "x" not in parts[0]:
            continue
        width_text, height_text = parts[0].split("x", 1)
        try:
            return int(width_text), int(height_text)
        except ValueError:
            continue
    return None
