# compiler.py - HIGH-END MULTI-PASS AI EDITOR (DUAL-STAGE ENGINE)
import os
import subprocess
import logging
import shutil
import sys
import random
import json
import glob
import time
import platform
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

# ==================== SETUP & CONFIG ====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("compiler")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

# Configuration
COMPUTE_MODE = os.getenv("COMPUTE_MODE", "auto").lower()
ENHANCEMENT_LEVEL = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
TRANSITION_DURATION = float(os.getenv("TRANSITION_DURATION", "1.0"))
TRANSITION_INTERVAL = int(os.getenv("TRANSITION_INTERVAL", "10"))
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "1080:1920")
REENCODE_CRF = os.getenv("REENCODE_CRF", "23")  # Changed from 18 to 23 for faster encoding
REENCODE_PRESET = os.getenv("REENCODE_PRESET", "veryfast")  # Changed from "slow" to "veryfast"

# AI Config
AI_FAST_MODE = os.getenv("AI_FAST_MODE", "no").lower() == "yes"
FACE_ENHANCEMENT = os.getenv("FACE_ENHANCEMENT", "yes").lower() == "yes"
USE_ADVANCED_ENGINE = os.getenv("USE_ADVANCED_ENGINE", "off").lower() == "on"

# Transformative Features Config
ADD_TEXT_OVERLAY = os.getenv("ADD_TEXT_OVERLAY", "yes").lower() == "yes"
ADD_COLOR_GRADING = os.getenv("ADD_COLOR_GRADING", "yes").lower() == "yes"
ADD_WATERMARK = os.getenv("ADD_WATERMARK", "no").lower() == "yes"
ADD_SPEED_RAMPING = os.getenv("ADD_SPEED_RAMPING", "yes").lower() == "yes"
FORCE_AUDIO_REMIX = os.getenv("FORCE_AUDIO_REMIX", "yes").lower() == "yes"

# Text Overlay Settings
TEXT_OVERLAY_TEXT = os.getenv("TEXT_OVERLAY_TEXT", "🔥 VIRAL")
TEXT_OVERLAY_POSITION = os.getenv("TEXT_OVERLAY_POSITION", "bottom")
TEXT_OVERLAY_STYLE = os.getenv("TEXT_OVERLAY_STYLE", "modern")

# Color Grading Settings
COLOR_FILTER = os.getenv("COLOR_FILTER", "cinematic")
COLOR_INTENSITY = float(os.getenv("COLOR_INTENSITY", "0.5"))

# Watermark Settings
WATERMARK_PATH = os.getenv("WATERMARK_PATH", "")
WATERMARK_POSITION = os.getenv("WATERMARK_POSITION", "topright")
WATERMARK_OPACITY = float(os.getenv("WATERMARK_OPACITY", "0.7"))

# Speed Ramping Settings
SPEED_VARIATION = float(os.getenv("SPEED_VARIATION", "0.15"))

TEMP_DIR = "temp"
OUTPUT_DIR = "merged_videos"
TOOLS_DIR = os.path.join(os.getcwd(), "tools")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def apply_zoom_effect(input_path: str, output_path: str):
    """Apply a slow Ken Burns style zoom effect."""
    logger.info(f"🔍 Applying Zoom Effect to: {input_path}")
    # Zoom in 10% over the duration
    vf = "zoompan=z='min(zoom+0.0015,1.5)':d=700:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920"
    
    encoder = _get_ffmpeg_encoder()
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", encoder, "-preset", "veryfast",
        "-c:a", "copy",
        output_path
    ]
    _run_command(cmd, check=True)

def add_text_overlay(input_path: str, output_path: str, text: str):
    """Add a professional text overlay."""
    logger.info(f"📝 Adding Text Overlay: {text}")
    
    # Determine position
    if TEXT_OVERLAY_POSITION == "top":
        y_pos = "h*0.15"
    elif TEXT_OVERLAY_POSITION == "middle":
        y_pos = "(h-text_h)/2"
    else: # bottom
        y_pos = "h*0.85"
        
    # Font settings (try to use a good font if available, else default)
    font_file = "arial.ttf"
    if platform.system() == "Windows":
        font_file = "C\\\\:/Windows/Fonts/arial.ttf"
    elif platform.system() == "Linux":
        font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
    # Drawtext filter
    drawtext = (
        f"drawtext=fontfile='{font_file}':text='{text}':"
        f"fontcolor=white:fontsize=80:x=(w-text_w)/2:y={y_pos}:"
        f"borderw=3:bordercolor=black:shadowx=2:shadowy=2"
    )
    
    encoder = _get_ffmpeg_encoder()
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", drawtext,
        "-c:v", encoder, "-preset", "veryfast",
        "-c:a", "copy",
        output_path
    ]
    _run_command(cmd, check=True)


def _run_command(cmd: List[str], check: bool = False, timeout: int = None) -> bool:
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {cmd[0]}")
        return False
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False

def _get_video_info(path: str) -> Dict:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", path
        ]
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        data = json.loads(result)
        stream = data["streams"][0]
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0))
        }
    except Exception:
        return {"width": 0, "height": 0, "duration": 0}

# ==================== ENGINE: HEAVY V2 (PYTORCH) ====================

# Heavy Engine Imports
try:
    from ai_engine import HeavyEditor
except ImportError:
    logger.warning("⚠️ ai_engine not found. AI enhancement will be disabled.")
    HeavyEditor = None

def enhance_video_auto(input_path: str, output_path: str, scale: int = 2) -> bool:
    """
    Automatic Engine Selection with FAST_MODE and ENHANCEMENT_LEVEL support.
    Levels:
    - basic: FFmpeg Lanczos Scaling (Fastest)
    - medium: RealESRGAN (Tile mode, No Face Enhance)
    - high: RealESRGAN (Full, Face Enhance if VRAM > 4GB)
    - ultra: RealESRGAN + GFPGAN (Full)
    """
    # 1. Check FAST_MODE (Strict)
    # FAST_MODE=yes -> Force FFmpeg, skip AI completely
    fast_mode = os.getenv("FAST_MODE", "no").lower() == "yes"
    
    # AI_FAST_MODE=yes -> Try AI, fallback if fails
    ai_fast_mode = os.getenv("AI_FAST_MODE", "no").lower() == "yes"
    
    enhancement_level = os.getenv("ENHANCEMENT_LEVEL", "high").lower()
    
    # Handle legacy enhancement levels
    if enhancement_level == "2x":
        enhancement_level = "high"
    elif enhancement_level == "4x":
        enhancement_level = "ultra"
    
    logger.info("=" * 60)
    logger.info(f"🚀 ENHANCEMENT ENGINE START")
    logger.info(f"⚡ FAST_MODE: {fast_mode}")
    logger.info(f"🤖 AI_FAST_MODE: {ai_fast_mode}")
    logger.info(f"🎚️ LEVEL: {enhancement_level}")
    logger.info("=" * 60)

    # FAST_MODE or Basic Level -> Skip AI
    if fast_mode or enhancement_level == "basic":
        logger.info("⚡ FAST_MODE enabled or Basic Level selected. Using FFmpeg Lanczos scaling.")
        return _ffmpeg_upscale(input_path, output_path, scale)

    # Try AI Enhancement
    try:
        if HeavyEditor is None:
            raise ImportError("ai_engine module missing")
            
        logger.info("🤖 Initializing AI Engine...")
        
        # Determine Face Enhance based on level
        face_enhance = (enhancement_level in ["high", "ultra"])
        
        editor = HeavyEditor(scale=scale, face_enhance=face_enhance)
        success = editor.process_video(input_path, output_path)
        
        if success:
            return True
        else:
            raise Exception("AI Processing returned False")
            
    except Exception as e:
        logger.error(f"❌ AI Enhancement Failed: {e}")
        
        if ai_fast_mode:
            logger.info("⚠️ AI_FAST_MODE enabled. Falling back to FFmpeg...")
            return _ffmpeg_upscale(input_path, output_path, scale)
        else:
            logger.error("❌ AI_FAST_MODE disabled. Aborting.")
            return False

def _ffmpeg_upscale(input_path, output_path, scale):
    """Fallback FFmpeg upscaling with sharpening."""
    logger.info("Running FFmpeg Upscale + Sharpen...")
    
    # Calculate target resolution
    info = _get_video_info(input_path)
    w, h = info['width'] * scale, info['height'] * scale
    
    # Lanczos scaling + Unsharp Mask
    vf = f"scale={w}:{h}:flags=lanczos,unsharp=5:5:1.0:5:5:0.0"
    
    encoder = _get_ffmpeg_encoder()
    preset = "veryfast"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", encoder, "-preset", preset,
        "-c:a", "copy",
        output_path
    ]
    
    return _run_command(cmd, check=True)





def _get_ffmpeg_encoder():
    """
    Detect if NVENC is available and working for hardware acceleration.
    """
    if COMPUTE_MODE == "cpu":
        return "libx264"
        
    try:
        # First check if NVENC is listed
        cmd = [FFMPEG_BIN, "-hide_banner", "-encoders"]
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        if "h264_nvenc" not in result:
            return "libx264"
        
        # Test if NVENC actually works by encoding a dummy frame
        test_cmd = [
            FFMPEG_BIN, "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        subprocess.check_output(test_cmd, stderr=subprocess.STDOUT, timeout=5)
        logger.info("🚀 NVENC (Hardware Acceleration) Detected and Working!")
        return "h264_nvenc"
    except:
        logger.info("ℹ️ NVENC not available or failed test. Using CPU encoding (libx264).")
        return "libx264"

def normalize_video(input_path: str, output_path: str):
    logger.info("📏 Normalizing video...")
    encoder = _get_ffmpeg_encoder()
    preset = "p4" if encoder == "h264_nvenc" else "veryfast"  # Use veryfast for CPU
    
    vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", encoder, "-preset", preset,
    ]
    
    # CRF is not supported by NVENC in the same way, use -cq for VBR or just bitrate
    if encoder == "libx264":
        cmd.extend(["-crf", "23"])  # Faster encoding, still good quality
    else:
        # NVENC settings for high quality
        cmd.extend(["-rc", "vbr", "-cq", "19", "-qmin", "19", "-qmax", "19"])

    cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2", output_path])
    
    _run_command(cmd, check=True)

# ==================== TRANSFORMATIVE FEATURES ====================

def add_text_overlay(input_path: str, output_path: str, text: str, position: str, style: str):
    """Add text overlay to video."""
    logger.info(f"📝 Adding text overlay: {text}")
    
    # Position mapping
    pos_map = {
        "top": "x=(w-text_w)/2:y=h*0.1",
        "center": "x=(w-text_w)/2:y=(h-text_h)/2",
        "bottom": "x=(w-text_w)/2:y=h*0.85"
    }
    
    # Style mapping
    if style == "modern":
        fontsize = 60
        fontcolor = "white"
        borderw = 3
        bordercolor = "black"
    elif style == "classic":
        fontsize = 50
        fontcolor = "yellow"
        borderw = 2
        bordercolor = "black"
    else:  # minimal
        fontsize = 45
        fontcolor = "white"
        borderw = 1
        bordercolor = "black@0.5"
    
    drawtext = f"drawtext=text='{text}':fontsize={fontsize}:fontcolor={fontcolor}:{pos_map.get(position, pos_map['bottom'])}:borderw={borderw}:bordercolor={bordercolor}"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", drawtext,
        "-c:a", "copy", output_path
    ]
    _run_command(cmd, check=True)

def apply_color_grading(input_path: str, output_path: str, filter_type: str, intensity: float):
    """Apply color grading filter to video."""
    logger.info(f"🎨 Applying color grading: {filter_type}")
    
    # Color filter presets
    filters = {
        "cinematic": f"eq=contrast=1.2:brightness=0.05:saturation=0.9,curves=all='0/0 0.5/{0.5-intensity*0.1} 1/1'",
        "vintage": f"eq=contrast=1.1:saturation=0.7,colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,curves=all='0/0.1 1/0.9'",
        "vibrant": f"eq=contrast=1.3:saturation={1.0+intensity}:brightness=0.02",
        "dark": f"eq=brightness=-0.1:contrast=1.4:saturation=0.8,curves=all='0/0 0.5/{0.4-intensity*0.1} 1/0.95'",
        "warm": f"colortemperature={6500+int(intensity*2000)}",
        "cool": f"colortemperature={6500-int(intensity*2000)}"
    }
    
    vf = filters.get(filter_type, filters["cinematic"])
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-c:a", "copy", output_path
    ]
    _run_command(cmd, check=True)

def add_watermark(input_path: str, output_path: str, watermark_path: str, position: str, opacity: float):
    """Add watermark overlay to video."""
    if not os.path.exists(watermark_path):
        logger.warning(f"⚠️ Watermark not found: {watermark_path}")
        # Copy input to output
        _run_command([FFMPEG_BIN, "-y", "-i", input_path, "-c", "copy", output_path], check=True)
        return
    
    logger.info(f"💧 Adding watermark: {position}")
    
    # Position mapping
    pos_map = {
        "topleft": "10:10",
        "topright": "main_w-overlay_w-10:10",
        "bottomleft": "10:main_h-overlay_h-10",
        "bottomright": "main_w-overlay_w-10:main_h-overlay_h-10"
    }
    
    overlay_pos = pos_map.get(position, pos_map["topright"])
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path, "-i", watermark_path,
        "-filter_complex", f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[logo];[0:v][logo]overlay={overlay_pos}",
        "-c:a", "copy", output_path
    ]
    _run_command(cmd, check=True)

def apply_speed_ramping(input_path: str, output_path: str, variation: float):
    """Apply random speed variations to video."""
    logger.info(f"⚡ Applying speed ramping: ±{variation*100}%")
    
    # Random speed between (1-variation) and (1+variation)
    speed = random.uniform(1.0 - variation, 1.0 + variation)
    
    # Use setpts for video speed
    vf = f"setpts={1/speed}*PTS"
    af = f"atempo={speed}"
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vf", vf,
        "-af", af,
        output_path
    ]
    _run_command(cmd, check=True)

# ==================== TRANSITIONS ====================

def create_transition_clip(seg_a: str, seg_b: str, output_path: str, trans_type: str, duration: float):
    info_a = _get_video_info(seg_a)
    dur_a = info_a['duration']
    start_a = max(0, dur_a - duration)
    
    tail_a = output_path.replace(".mp4", "_tailA.mp4")
    _run_command([FFMPEG_BIN, "-y", "-ss", str(start_a), "-i", seg_a, "-t", str(duration), "-c", "copy", tail_a])
    
    head_b = output_path.replace(".mp4", "_headB.mp4")
    _run_command([FFMPEG_BIN, "-y", "-i", seg_b, "-t", str(duration), "-c", "copy", head_b])
    
    filter_str = f"[0:v][1:v]xfade=transition={trans_type}:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"
    if trans_type == "zoom":
        filter_str = f"[0:v][1:v]xfade=transition=circleopen:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", tail_a, "-i", head_b,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", # Transitions are short, keep CPU for stability/compatibility
        "-c:a", "aac",
        output_path
    ]
    _run_command(cmd)
    
    if os.path.exists(tail_a): os.remove(tail_a)
    if os.path.exists(head_b): os.remove(head_b)

def compile_with_transitions(input_video: Path, title: str) -> Path:
    import audio_processing
    
    input_path = os.path.abspath(str(input_video))
    job_id = f"job_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Add timestamp to ensure unique output files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_output = os.path.join(OUTPUT_DIR, f"final_{title}_{timestamp}.mp4")
    
    try:
        # Get video info for smart processing
        video_info = _get_video_info(input_path)
        duration = video_info.get('duration', 0)
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        
        logger.info(f"📊 Video: {width}x{height}, {duration:.1f}s")
        
        # 1. AI Enhancement (skip on CPU)
        logger.info("✨ Step 1: AI Enhancement")
        enhanced_video = os.path.join(job_dir, "enhanced.mp4")
        
        scale = 2
        if ENHANCEMENT_LEVEL and ENHANCEMENT_LEVEL[0].isdigit():
            scale = int(ENHANCEMENT_LEVEL[0])
            
        success = enhance_video_auto(input_path, enhanced_video, scale)
        
        if not success:
            logger.warning("⚠️ Enhancement failed. Using original video.")
            enhanced_video = input_path
        
        # 2. Smart Normalization (skip if already correct specs)
        needs_normalization = (width != 1080 or height != 1920)
        
        if needs_normalization:
            logger.info("✨ Step 2: Normalization (required)")
            norm_video = os.path.join(job_dir, "normalized.mp4")
            normalize_video(enhanced_video, norm_video)
        else:
            logger.info("⚡ Step 2: Normalization (skipped - already correct specs)")
            # Use stream copy for speed
            norm_video = os.path.join(job_dir, "normalized.mp4")
            _run_command([
                FFMPEG_BIN, "-y", "-i", enhanced_video,
                "-c", "copy", norm_video
            ], check=True)
        
        # 2.5 Apply Transformative Features
        current_video = norm_video
        
        # Text Overlay
        if ADD_TEXT_OVERLAY:
            try:
                logger.info("✨ Step 2.5: Text Overlay")
                text_video = os.path.join(job_dir, "text_overlay.mp4")
                add_text_overlay(current_video, text_video, TEXT_OVERLAY_TEXT)
                if os.path.exists(text_video):
                    current_video = text_video
                else:
                    logger.warning("⚠️ Text overlay failed, using previous video")
            except Exception as e:
                logger.warning(f"⚠️ Text overlay error: {e}, skipping")
        
        # Color Grading
        if ADD_COLOR_GRADING:
            try:
                logger.info("✨ Step 2.6: Color Grading")
                color_video = os.path.join(job_dir, "color_graded.mp4")
                apply_color_grading(current_video, color_video, COLOR_FILTER, COLOR_INTENSITY)
                if os.path.exists(color_video):
                    current_video = color_video
                else:
                    logger.warning("⚠️ Color grading failed, using previous video")
            except Exception as e:
                logger.warning(f"⚠️ Color grading error: {e}, skipping")
        
        # Watermark
        if ADD_WATERMARK and WATERMARK_PATH:
            try:
                logger.info("✨ Step 2.7: Watermark")
                watermark_video = os.path.join(job_dir, "watermarked.mp4")
                add_watermark(current_video, watermark_video, WATERMARK_PATH, WATERMARK_POSITION, WATERMARK_OPACITY)
                if os.path.exists(watermark_video):
                    current_video = watermark_video
                else:
                    logger.warning("⚠️ Watermark failed, using previous video")
            except Exception as e:
                logger.warning(f"⚠️ Watermark error: {e}, skipping")
        
        # Speed Ramping
        if ADD_SPEED_RAMPING:
            try:
                logger.info("✨ Step 2.8: Speed Ramping")
                speed_video = os.path.join(job_dir, "speed_ramped.mp4")
                apply_speed_ramping(current_video, speed_video, SPEED_VARIATION)
                if os.path.exists(speed_video):
                    current_video = speed_video
                else:
                    logger.warning("⚠️ Speed ramping failed, using previous video")
            except Exception as e:
                logger.warning(f"⚠️ Speed ramping error: {e}, skipping")
        
        # Update norm_video to current_video for next steps
        norm_video = current_video
        
        # 3. Smart Transitions (skip for short videos)
        skip_transitions = duration < 30  # Skip transitions for videos <30s
        
        if skip_transitions:
            logger.info("⚡ Step 3: Transitions (skipped - video too short)")
            # Apply Zoom for static/short videos to make them transformative
            if ADD_SPEED_RAMPING: # Using as proxy for "add motion"
                 logger.info("🔍 Applying Zoom Effect (Transformative Motion)")
                 zoomed_video = os.path.join(job_dir, "zoomed.mp4")
                 apply_zoom_effect(norm_video, zoomed_video)
                 merged_video = zoomed_video
            else:
                 merged_video = norm_video
        else:
            logger.info("✨ Step 3: Segmentation")
            seg_pattern = os.path.join(job_dir, "seg_%03d.mp4")
            cmd_split = [
                FFMPEG_BIN, "-y", "-i", norm_video,
                "-c", "copy", "-f", "segment", "-segment_time", str(TRANSITION_INTERVAL),
                "-reset_timestamps", "1", seg_pattern
            ]
            _run_command(cmd_split, check=True)
            segments = sorted(glob.glob(os.path.join(job_dir, "seg_*.mp4")))
            
            if len(segments) < 2:
                logger.info("   Video too short for transitions.")
                # Apply Zoom here too if transitions failed
                if ADD_SPEED_RAMPING:
                     zoomed_video = os.path.join(job_dir, "zoomed.mp4")
                     apply_zoom_effect(norm_video, zoomed_video)
                     merged_video = zoomed_video
                else:
                     merged_video = norm_video
            else:
                # 4. Transitions
                logger.info("✨ Step 4: Transitions")
                final_segments = []
                transitions = ["fade", "slideleft", "slideright", "wipeleft", "wiperight", "circleopen", "circleclose", "zoom"]
                
                seg0 = segments[0]
                dur0 = _get_video_info(seg0)['duration']
                trim0 = os.path.join(job_dir, "final_seg_000.mp4")
                _run_command([FFMPEG_BIN, "-y", "-i", seg0, "-t", str(max(0, dur0 - TRANSITION_DURATION)), "-c", "copy", trim0])
                final_segments.append(trim0)
                
                for i in range(len(segments) - 1):
                    seg_curr = segments[i]
                    seg_next = segments[i+1]
                    trans_type = random.choice(transitions)
                    trans_path = os.path.join(job_dir, f"final_trans_{i}.mp4")
                    create_transition_clip(seg_curr, seg_next, trans_path, trans_type, TRANSITION_DURATION)
                    final_segments.append(trans_path)
                    
                    is_last = (i + 1) == (len(segments) - 1)
                    dur_next = _get_video_info(seg_next)['duration']
                    start_trim = TRANSITION_DURATION
                    end_trim = 0 if is_last else TRANSITION_DURATION
                    keep_dur = max(0, dur_next - start_trim - end_trim)
                    
                    body_next = os.path.join(job_dir, f"final_seg_{i+1:03d}.mp4")
                    _run_command([
                        FFMPEG_BIN, "-y", "-ss", str(start_trim), "-i", seg_next,
                        "-t", str(keep_dur), "-c", "copy", body_next
                    ])
                    final_segments.append(body_next)

                # 5. Merge
                logger.info("✨ Step 5: Merging")
                list_file = os.path.join(job_dir, "merge_list.txt")
                with open(list_file, "w") as f:
                    for p in final_segments:
                        f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
                
                merged_video = os.path.join(job_dir, "merged_video.mp4")
                _run_command([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_video], check=True)

        # 6. Smart Audio Remix (ALWAYS RUN for transformative content)
        # We removed the skip logic. Audio processing now handles looping.
        if FORCE_AUDIO_REMIX:
            logger.info("✨ Step 6: Audio Remix (forced - transformative mode)")
        else:
            logger.info("✨ Step 6: Audio Remix")
            
        remixed_audio = os.path.join(job_dir, "remixed.wav")
        audio_processing.heavy_remix(merged_video, remixed_audio)
        
        # 7. Final Assembly
        logger.info("✨ Step 7: Final Assembly")
        
        encoder = _get_ffmpeg_encoder()
        preset = "p4" if encoder == "h264_nvenc" else REENCODE_PRESET
        
        cmd_final = [
            FFMPEG_BIN, "-y", "-i", merged_video, "-i", remixed_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", encoder, "-preset", preset,
            "-shortest", final_output
        ]
        
        if encoder == "libx264":
            cmd_final.extend(["-crf", REENCODE_CRF])
        else:
            cmd_final.extend(["-rc", "vbr", "-cq", "19", "-qmin", "19", "-qmax", "19"])
            
        _run_command(cmd_final, check=True)
        
        if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
            # Final Quality Check
            return Path(verify_and_fix_output(final_output))
        else:
            raise Exception("Output creation failed")
            
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        return None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

# Legacy function for compilation support
def compile_batch_with_transitions(video_files: List[str], output_filename: str) -> Optional[str]:
    """
    Compiles multiple video files into one with transitions, normalization, and audio remixing.
    Replaces the old streamcopy merge.
    """
    import audio_processing
    
    # 1. Setup Paths
    if os.path.isabs(output_filename):
        final_output = output_filename
    else:
        final_output = os.path.abspath(os.path.join(OUTPUT_DIR, output_filename))
        
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    job_id = f"batch_{int(time.time())}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting Batch Compilation for {len(video_files)} videos...")
        
        # 2. Normalize Inputs
        logger.info("✨ Step 1: Normalizing Inputs")
        normalized_clips = []
        for i, vid in enumerate(video_files):
            norm_path = os.path.join(job_dir, f"norm_{i:03d}.mp4")
            # Ensure absolute path for input
            abs_input = os.path.abspath(vid)
            if not os.path.exists(abs_input):
                logger.warning(f"⚠️ Input file not found: {abs_input}")
                continue
                
            normalize_video(abs_input, norm_path)
            if os.path.exists(norm_path):
                normalized_clips.append(norm_path)
            else:
                logger.error(f"❌ Failed to normalize: {vid}")

    except Exception as e:
        logger.error(f"Batch Compilation Failed: {e}")
        return None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
