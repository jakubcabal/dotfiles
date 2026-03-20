#!/usr/bin/env python3
# Copyright (c) 2026 Jakub Cabal <jakubcabal@gmail.com>
# SPDX-License-Identifier: MIT

"""FFmpeg VAAPI Encoder Script by Jakub Cabal"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys

DEVICE = "/dev/dri/renderD128"
TOOLS = ["ffmpeg", "mkvpropedit", "ffprobe"]


def fmt_size(bytes_val):
    """Format file size in human-readable format."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(bytes_val) < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PiB"


def fmt_duration(seconds):
    """Format duration as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_input_info(filepath):
    """Get input file information using ffprobe."""
    if not os.path.isfile(filepath):
        print(f"Error: Input file not found: {filepath}")
        sys.exit(1)

    info = {"name": os.path.basename(filepath), "size": os.path.getsize(filepath)}

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", filepath],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        fmt = data.get("format", {})

        info["duration"] = fmt_duration(float(fmt.get("duration", 0)))

        streams = data.get("streams", [])
        video = [s for s in streams if s.get("codec_type") == "video"]
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        subs = [s for s in streams if s.get("codec_type") == "subtitle"]

        info["video"] = video[0] if video else {}
        info["audio_count"] = len(audio)
        info["subs_count"] = len(subs)
    except Exception:
        info["duration"] = "??:??:??"
        info["video"] = {}
        info["audio_count"] = 0
        info["subs_count"] = 0

    return info


def print_input_info(info):
    """Print input file information."""
    video = info.get("video", {})
    width = video.get("width", "?")
    height = video.get("height", "?")
    codec = video.get("codec_name", "?")
    
    print(f"\nInput: {info['name']} | {fmt_size(info['size'])} | {info.get('duration', '??')}")
    print(f"Video: {width}x{height} ({codec}) | Audio: {info.get('audio_count', 0)} | Subs: {info.get('subs_count', 0)}")


def check_dependencies():
    """Check if required tools and VAAPI are available."""
    missing = [t for t in TOOLS if not shutil.which(t)]
    if missing:
        print(f"Error: Missing: {' '.join(missing)}")
        sys.exit(1)
    
    if not os.path.exists(DEVICE):
        print(f"Error: VAAPI device not found: {DEVICE}")
        sys.exit(1)


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="ffmpeg_vaapi_enc.py",
        description="FFmpeg VAAPI Encoder Script by Jakub Cabal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i input.mkv
  %(prog)s -i input.mkv -o output.mkv -e hevc -q 18
  %(prog)s -i input.mkv -e av1 -q 28 -a aac -ab 256K

Encoders: h264, hevc, av1
Audio: copy, aac, ac3, eac3, flac, opus, mp3
Quality: 18-28 (lower = better quality)
"""
    )

    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", default="output.mkv", help="Output filename")
    parser.add_argument(
        "-e", "--encoder", default="h264",
        choices=["h264", "hevc", "av1"],
        help="Video encoder (default: h264)"
    )
    parser.add_argument("-q", "--qp", type=int, default=22, help="Quality parameter 1-51")
    parser.add_argument("-s", "--scale", help="Video resolution (e.g., 1920x1080, 1280x-1)")
    parser.add_argument(
        "-a", "--audio", default="copy",
        choices=["copy", "aac", "ac3", "eac3", "flac", "opus", "mp3"],
        help="Audio codec (default: copy)"
    )
    parser.add_argument("-ab", "--audio-bitrate", default="320K", help="Audio bitrate")

    return parser


def main():
    """Main entry point."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        create_parser().print_help()
        sys.exit(1)

    args = create_parser().parse_args()

    # Expand ~ in input path
    args.input = os.path.expanduser(args.input)

    # Validate QP
    if args.qp < 1 or args.qp > 51:
        print(f"Error: Invalid QP '{args.qp}'. Must be between 1 and 51.")
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Get and print input info
    info = get_input_info(args.input)
    print_input_info(info)

    # Build video filter
    video_filter = f"scale_vaapi={args.scale}," if args.scale else ""
    video_filter += "format=nv12,hwupload"

    # Encoder mapping
    encoders = {
        "h264": "h264_vaapi",
        "hevc": "hevc_vaapi",
        "av1": "av1_vaapi"
    }

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-hwaccel", "vaapi", "-vaapi_device", DEVICE,
        "-i", args.input,
        "-map", "0:v", "-map", "0:a?", "-map", "0:s?",
        "-vf", video_filter,
        "-c:v", encoders[args.encoder],
        "-qp", str(args.qp),
        "-c:s", "copy"
    ]

    # Audio codec
    if args.audio == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", args.audio, "-b:a", args.audio_bitrate]

    # Metadata and output
    cmd += ["-map_metadata", "0", "-y", args.output]

    # Print settings and command
    audio_info = f"{args.audio} ({args.audio_bitrate})" if args.audio != "copy" else args.audio
    print(f"Output: {args.output} | Encoder: {encoders[args.encoder]} | QP: {args.qp} | Audio: {audio_info}")
    if args.scale:
        print(f"Scale: {args.scale}")
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run ffmpeg and display progress
    run_ffmpeg(cmd)
    
    print("\nEncoding completed!")
    
    # Post-process
    if os.path.isfile(args.output):
        subprocess.run(["mkvpropedit", args.output, "--add-track-statistics-tags"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Summary
        output_size = os.path.getsize(args.output)
        ratio = output_size * 100 / info["size"] if info["size"] > 0 else 0
        print(f"\nDone! Output: {fmt_size(output_size)} ({ratio:.1f}% of original)\n")


def run_ffmpeg(cmd):
    """Run ffmpeg command and display progress line."""
    # Start ffmpeg with stderr capture
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nInterrupted, terminating ffmpeg...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Stream only frame= progress line
    try:
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                # Show only frame= progress line
                if line.startswith("frame="):
                    print(f"\r{line.strip()}", end="", flush=True)
        
        print()  # Final newline
        
        if proc.returncode != 0:
            print(f"Error: ffmpeg exited with code {proc.returncode}")
            sys.exit(1)
    finally:
        # Restore default signal handler
        signal.signal(signal.SIGINT, signal.default_int_handler)


if __name__ == "__main__":
    main()
