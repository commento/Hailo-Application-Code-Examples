#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from functools import partial
from types import SimpleNamespace
import numpy as np
import argparse
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoInfer
from object_detection_post_process import inference_result_handler
from common.toolbox import (
    init_input_source,
    get_labels,
    load_json_file,
    preprocess,
    visualize,
    FrameRateTracker,
    resolve_net_arg,
    resolve_input_arg,
    resolve_output_resolution_arg,
    list_networks,
    list_inputs
)
from aura_runtime import (
    AudioAnalyzer,
    AuraPostProcessor,
    RisingAuraRenderer,
    visualize_with_aura_recording,
)

APP_NAME = Path(__file__).stem

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run object detection with optional tracking and performance measurement.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-n", "--net",
        type=str,
        help=(
            "- A local HEF file path\n"
            "    → uses the specified HEF directly.\n"
            "- A model name (e.g., yolov8n)\n"
            "    → automatically downloads & resolves the correct HEF for your device.\n"
            "      Use --list-nets to see the available nets."
        )    
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help=(
            "Input source. Examples:\n"
            "  - Local path: 'bus.jpg', 'video.mp4', 'images_dir/'\n"
            "  - Special:    'camera'\n"
            "  - Named resource (without extension), e.g. 'bus'.\n"
            "    If a named resource is used, it will be downloaded automatically\n"
            "    if not already available. Use --list-inputs to see the options."
        )
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Number of images per batch."
    )
    parser.add_argument(
        "-l", "--labels",
        type=str,
        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
        help="Path to label file (e.g., coco.txt). If not set, default COCO labels will be used."
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the visualized stream output to disk."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save result images or video."
    )
    parser.add_argument(
        "-f", "--framerate",
        type=float,
        default=None,
        help=(
            "[Camera only] Override the camera input framerate.\n"
            "Example: -f 10.0"
        )
    )
    parser.add_argument(
        "--draw-trail",
        action="store_true",
        help=(
            "[Tracking only] Draw motion trails of tracked objects.\n"
            "Uses the last 30 positions from the tracker history."
        )
    )
    parser.add_argument(
        "--aura",
        action="store_true",
        help="Draw an upward-moving aura around detected people, gated by live audio input."
    )
    parser.add_argument(
        "--aura-audio-device",
        type=str,
        default=None,
        help="Audio input device for the aura gate. Use a sounddevice device id/name; default uses system input."
    )
    parser.add_argument(
        "--aura-audio-threshold",
        type=float,
        default=0.004,
        help="Minimum input audio RMS before the aura appears."
    )
    parser.add_argument(
        "--aura-audio-scale",
        type=float,
        default=20.0,
        help="Multiplier used to convert audio level into aura intensity."
    )
    parser.add_argument(
        "--aura-radius",
        type=int,
        default=120,
        help="Base radius of the aura around each detected person."
    )
    parser.add_argument(
        "--aura-alpha",
        type=float,
        default=0.58,
        help="Aura overlay opacity."
    )
    parser.add_argument(
        "--aura-background-dim",
        type=float,
        default=0.0,
        help="Background dim amount while the aura is active. Defaults to 0.0 to keep the live image bright."
    )
    parser.add_argument(
        "--aura-edge-warp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fisheye-style distortion on the screen edges while the aura is active."
    )
    parser.add_argument(
        "--aura-edge-warp-strength",
        type=float,
        default=0.34,
        help="Strength of the aura edge fisheye distortion."
    )
    parser.add_argument(
        "--aura-debug-boxes",
        action="store_true",
        help="Draw person tracking boxes on top of the aura output."
    )
    parser.add_argument(
        "--record-performance",
        action="store_true",
        help="Record the aura performance to an MP4 with microphone audio using ffmpeg."
    )
    parser.add_argument(
        "--recording-output",
        type=str,
        default=None,
        help="ffmpeg output path pattern, e.g. output/aura_%Y%m%d_%H%M%S.mp4."
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable path."
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=48000,
        help="Audio sample rate used for aura analysis and recording."
    )
    parser.add_argument(
        "--audio-block-size",
        type=int,
        default=512,
        help="Audio callback block size used for aura analysis and recording."
    )
    display_group = parser.add_mutually_exclusive_group(required=False)
    display_group.add_argument(
        "-cr","--camera-resolution",
        type=str,
        choices=["sd", "hd", "fhd"],
        help="(Camera only) Input resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080)."
    )
    display_group.add_argument(
        "-or","--output-resolution",
        nargs="+",
        type=str,
        help=(
            "(Camera only) Output resolution. Use: 'sd', 'hd', 'fhd', "
            "or custom size like '--output-resolution 1920 1080'."
        )
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking across frames."
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Enable FPS measurement and display."
    )
    parser.add_argument(
        "--list-nets",
        action="store_true",
        help="List supported nets for this app and exit"
    )
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="List predefined sample inputs for this app and exit."
    )
    args = parser.parse_args()

    # Handle --list-nets and exit
    if args.list_nets:
        list_networks(APP_NAME)
        sys.exit(0)

    # Handle --list-inputs and exit
    if args.list_inputs:
        list_inputs(APP_NAME)
        sys.exit(0)

    args.net = resolve_net_arg(APP_NAME, args.net, ".")
    args.input = resolve_input_arg(APP_NAME, args.input)
    args.output_resolution = resolve_output_resolution_arg(args.output_resolution)

    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    return args



def run_inference_pipeline(net, input, batch_size, labels, output_dir,
          save_stream_output=False, camera_resolution=None, output_resolution=None,
          enable_tracking=False, show_fps=False, framerate=None, draw_trail=False,
          aura=False, aura_audio_device=None, aura_audio_threshold=0.004,
          aura_audio_scale=20.0, aura_radius=120, aura_alpha=0.58,
          aura_background_dim=0.0, aura_edge_warp=False, aura_edge_warp_strength=0.34,
          aura_debug_boxes=False,
          record_performance=False, recording_output=None, ffmpeg_bin="ffmpeg",
          audio_sample_rate=48000, audio_block_size=512) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    labels = get_labels(labels)
    config_data = load_json_file("config.json")

    # Initialize input source from string: "camera", video file, or image folder.
    cap, images = init_input_source(input, batch_size, camera_resolution)
    tracker = None
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking or aura:
        # load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    audio_analyzer = None
    if aura:
        audio_device = _parse_audio_device(aura_audio_device)
        audio_analyzer = AudioAnalyzer(audio_sample_rate, audio_block_size, audio_device)
        aura_renderer = RisingAuraRenderer(
            aura_radius=aura_radius,
            aura_alpha=aura_alpha,
            background_dim=aura_background_dim,
            audio_threshold=aura_audio_threshold,
            audio_scale=aura_audio_scale,
            edge_warp=aura_edge_warp,
            edge_warp_strength=aura_edge_warp_strength,
        )
        post_process_callback_fn = AuraPostProcessor(
            labels=labels,
            config_data=config_data,
            tracker=tracker,
            audio=audio_analyzer,
            renderer=aura_renderer,
            draw_boxes=aura_debug_boxes,
        )
    else:
        post_process_callback_fn = partial(
            inference_result_handler, labels=labels,
            config_data=config_data, tracker=tracker, draw_trail=draw_trail
        )

    hailo_inference = HailoInfer(net, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(images, cap, framerate, batch_size, input_queue, width, height)
    )
    if aura:
        postprocess_thread = threading.Thread(
            target=visualize_with_aura_recording,
            kwargs={
                "output_queue": output_queue,
                "cap": cap,
                "save_stream_output": save_stream_output,
                "output_dir": output_dir,
                "callback": post_process_callback_fn,
                "audio": audio_analyzer,
                "fps_tracker": fps_tracker,
                "output_resolution": output_resolution,
                "framerate": framerate,
                "record_audio_video": record_performance or save_stream_output,
                "ffmpeg_bin": ffmpeg_bin,
                "recording_output": recording_output,
            }
        )
    else:
        postprocess_thread = threading.Thread(
            target=visualize, args=(output_queue, cap, save_stream_output,
                                    output_dir, post_process_callback_fn,
                                    fps_tracker, output_resolution)
        )
    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )

    if audio_analyzer is not None:
        audio_analyzer.start()

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    if audio_analyzer is not None:
        audio_analyzer.stop()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")
    if save_stream_output or record_performance or input.lower() != "camera":
        logger.success(f'Results have been saved in {output_dir}')


def _parse_audio_device(device):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def infer(hailo_inference, input_queue, output_queue):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))


def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()

    run_inference_pipeline(args.net, args.input, args.batch_size, args.labels,
          args.output_dir, args.save_stream_output, args.camera_resolution,
          args.output_resolution, args.track, args.show_fps, args.framerate, args.draw_trail,
          args.aura, args.aura_audio_device, args.aura_audio_threshold,
          args.aura_audio_scale, args.aura_radius, args.aura_alpha,
          args.aura_background_dim, args.aura_edge_warp,
          args.aura_edge_warp_strength, args.aura_debug_boxes,
          args.record_performance, args.recording_output, args.ffmpeg_bin,
          args.audio_sample_rate, args.audio_block_size)




if __name__ == "__main__":
    main()
