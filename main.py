from dotenv import load_dotenv
load_dotenv()

import os
import logging
import asyncio
import shutil
import sys
import re
import time
import subprocess
import csv
import json
import platform
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut
from urllib.parse import urlparse
from pathlib import Path

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_DOMAINS = ["youtube.com", "youtu.be", "instagram.com", "twitter.com", "x.com", "facebook.com"]
UPLOAD_LOG = os.getenv("VIDEO_LOG_FILE", "upload_log.csv")
COMPILATION_BATCH_SIZE = int(os.getenv("COMPILATION_BATCH_SIZE", "6"))

user_sessions = {}

# ==================== AUTO-INSTALL & SETUP ====================

# ==================== AUTO-INSTALL & SETUP ====================

def detect_hardware():
    """
    Detect hardware capabilities BEFORE loading any heavy libraries.
    Returns: dict with 'has_gpu', 'gpu_name', 'vram_gb', 'cuda_available'
    """
    hardware_info = {
        'has_gpu': False,
        'gpu_name': 'CPU',
        'vram_gb': 0,
        'cuda_available': False
    }
    
    try:
        # Try to detect NVIDIA GPU without importing torch
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(',')
            hardware_info['has_gpu'] = True
            hardware_info['gpu_name'] = gpu_info[0].strip()
            hardware_info['vram_gb'] = int(gpu_info[1].strip().split()[0]) / 1024
            logger.info(f"🎮 GPU Detected: {hardware_info['gpu_name']} ({hardware_info['vram_gb']:.1f} GB VRAM)")
    except:
        logger.info("ℹ️ No NVIDIA GPU detected. Running in CPU mode.")
    
    # Now check if PyTorch can use CUDA (only if we detected a GPU)
    if hardware_info['has_gpu']:
        try:
            import torch
            hardware_info['cuda_available'] = torch.cuda.is_available()
            if not hardware_info['cuda_available']:
                logger.warning("⚠️ GPU detected but PyTorch CUDA not available. Install CUDA-enabled PyTorch.")
        except ImportError:
            logger.warning("⚠️ PyTorch not installed. GPU will not be used.")
    
    return hardware_info

def check_and_update_env():
    """
    Auto-updates .env file with missing keys and smart defaults.
    Requirement 7: Auto-Check for Missing .env Keys
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        logger.warning("⚠️ .env file not found. Creating template...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("""# ==================== CORE SETTINGS ====================
# REQUIRED: Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE

# REQUIRED: Get your API key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# ==================== PERFORMANCE ====================
# FAST_MODE: yes = FFmpeg only (10x faster), no = AI enhancement
FAST_MODE=no
AI_FAST_MODE=yes
COMPUTE_MODE=auto
ENHANCEMENT_LEVEL=medium

# ==================== TRANSFORMATIVE FEATURES ====================
ADD_TEXT_OVERLAY=yes
TEXT_OVERLAY_TEXT=🔥 VIRAL
TEXT_OVERLAY_POSITION=bottom
TEXT_OVERLAY_STYLE=modern

ADD_COLOR_GRADING=yes
COLOR_FILTER=cinematic
COLOR_INTENSITY=0.5

ADD_SPEED_RAMPING=yes
SPEED_VARIATION=0.15

FORCE_AUDIO_REMIX=yes

# ==================== COMPILATION ====================
COMPILATION_BATCH_SIZE=6
SEND_TO_YOUTUBE=off
TARGET_RESOLUTION=1080:1920
TRANSITION_DURATION=1.0
TRANSITION_INTERVAL=10
""")
        logger.info("✅ Created .env template. Please update TELEGRAM_BOT_TOKEN and GEMINI_API_KEY!")
        
    # Load current env content
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if critical keys are missing
    if "YOUR_BOT_TOKEN_HERE" in content or "YOUR_GEMINI_API_KEY_HERE" in content:
        logger.error("❌ CRITICAL: Please update .env file with your actual API keys!")
        logger.error("   - TELEGRAM_BOT_TOKEN (from @BotFather)")
        logger.error("   - GEMINI_API_KEY (from https://aistudio.google.com/app/apikey)")
        # Don't exit, let user fix it
        
    updates = []
    
    # Define required keys and defaults
    required_keys = {
        "FAST_MODE": "no",
        "AI_FAST_MODE": "yes",
        "ENHANCEMENT_LEVEL": "medium",
        "COMPUTE_MODE": "auto",
        "TRANSITION_INTERVAL": "10",
        "FORCE_AUDIO_REMIX": "yes",
        "ADD_TEXT_OVERLAY": "yes",
        "ADD_SPEED_RAMPING": "yes"
    }
    
    for key, default in required_keys.items():
        if key not in os.environ and f"{key}=" not in content:
            logger.info(f"➕ Auto-adding missing key: {key}={default}")
            updates.append(f"\n# Auto-added by Smart Installer\n{key}={default}")
            os.environ[key] = default # Set in current session too
            
    if updates:
        with open(env_path, "a", encoding="utf-8") as f:
            for line in updates:
                f.write(line)
        logger.info("✅ .env file updated with missing keys.")

def ensure_requirements_and_tools():
    """
    Smart Installation System:
    1. Detects Hardware (GPU/CPU) FIRST
    2. Checks FAST_MODE preference
    3. Installs ONLY necessary dependencies
    """
    logger.info("🔧 Starting Smart Startup Checks...")
    
    # 0. Auto-Update .env
    check_and_update_env()
    
    # 1. Detect Hardware BEFORE any heavy imports
    hardware = detect_hardware()
    
    # 2. Check FAST_MODE
    fast_mode_env = os.getenv("FAST_MODE")
    if fast_mode_env is None:
        fast_mode_env = os.getenv("AI_FAST_MODE", "no")
        
    fast_mode = fast_mode_env.lower().strip() == "yes"
    
    # Update env for other scripts to see
    os.environ["FAST_MODE"] = "yes" if fast_mode else "no"
    
    if os.getenv("ENHANCEMENT_LEVEL") is None:
        os.environ["ENHANCEMENT_LEVEL"] = "medium"

    logger.info(f"⚡ FAST_MODE: {fast_mode} (Source: {fast_mode_env})")
    
    # 3. Smart Dependency Installation
    if fast_mode:
        logger.info("⚡ FAST_MODE enabled. Skipping heavy AI dependencies (scipy, torch, etc.)")
        # Install only lightweight dependencies
        lightweight_deps = ["python-telegram-bot", "python-dotenv", "requests", "google-generativeai"]
        try:
            for dep in lightweight_deps:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
    else:
        # Full installation for AI mode
        if os.path.exists("requirements.txt"):
            try:
                logger.info("📦 Installing full dependencies (AI mode)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error(f"❌ Failed to install requirements: {e}")

        # Install AI tools only if GPU detected or user wants AI
        try:
            logger.info("🔧 Checking AI Models & Tools...")
            subprocess.check_call([sys.executable, "tools-install.py"])
        except Exception as e:
            logger.error(f"❌ Tools installation failed: {e}")
        
    logger.info("✅ Startup checks complete.")

# Run setup BEFORE imports that might need them
ensure_requirements_and_tools()

# Conditional imports based on FAST_MODE
fast_mode = os.getenv("FAST_MODE", "no").lower() == "yes"

try:
    import downloader
    import uploader
    from compiler import compile_with_transitions, compile_batch_with_transitions
    
    # Only import audio_processing if NOT in FAST_MODE (it requires scipy)
    if not fast_mode:
        import audio_processing
    else:
        logger.info("ℹ️ Skipping audio_processing import (FAST_MODE enabled)")
        audio_processing = None
        
except ImportError as e:
    logger.error(f"Critical Import Error: {e}")
    sys.exit(1)

# ==================== UTILS ====================

def _ensure_log_header():
    if not os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "file_path", "yt_link", "title"])

def log_video(file_path: str, yt_link: str, title: str):
    _ensure_log_header()
    with open(UPLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), file_path, yt_link, title])

def total_uploads() -> int:
    if not os.path.exists(UPLOAD_LOG):
        return 0
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        return max(0, len(rows) - 1)

def last_n_filepaths(n: int) -> list:
    """Get the last N video file paths from the upload log, filtered by recency."""
    if not os.path.exists(UPLOAD_LOG):
        return []
    
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter by timestamp - only videos from last 24 hours
    from datetime import datetime, timedelta
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    recent_rows = []
    for r in rows:
        try:
            timestamp_str = r.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_rows.append(r)
        except:
            # If timestamp parsing fails, skip this row
            continue
    
    # Get last N from recent rows
    subset = recent_rows[-n:]
    paths = [r.get("file_path") for r in subset if r.get("file_path")]
    
    # Return only paths that exist
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    
    logger.info(f"📊 Found {len(valid_paths)} recent videos for compilation (last 24h)")
    return valid_paths

async def safe_reply(update: Update, text: str):
    for _ in range(3):
        try:
            if update.message:
                await update.message.reply_text(text)
            return
        except (NetworkError, TimedOut) as e:
            logger.warning("🛑 Reply failed: %s. Retrying...", e)
            await asyncio.sleep(2)
    logger.error("❌ Failed to send message after retries.")

async def safe_video_reply(update: Update, video_path: str, caption: str = None):
    """
    Robustly reply with a video, handling timeouts and retries.
    """
    for attempt in range(1, 4):
        try:
            if update.message:
                # read_timeout/write_timeout kwargs are supported in send_video (which reply_video wraps)
                # We set a very high timeout for large file uploads
                await update.message.reply_video(
                    video_path, 
                    caption=caption, 
                    read_timeout=600, 
                    write_timeout=600,
                    connect_timeout=60,
                    pool_timeout=60
                )
            return
        except (NetworkError, TimedOut) as e:
            logger.warning(f"🛑 Video reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"❌ Video reply error: {e}")
            break
            
    logger.error("❌ Failed to send video after retries.")
    await safe_reply(update, "❌ Failed to send video due to network timeout.")

def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(allowed in domain for allowed in ALLOWED_DOMAINS)
    except: return False

def _sanitize_title(title: str) -> str:
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = clean.replace(' ', '_')
    return clean[:50]

def _get_hashtags(text: str) -> str:
    link_count = len(re.findall(r'https?://', text))
    if link_count > 1:
        return os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").strip()
    return os.getenv("DEFAULT_HASHTAGS_SHORTS", "").strip()

def _detect_hardware():
    logger.info("🔍 Detecting hardware...")
    try:
        subprocess.check_output("nvidia-smi", stderr=subprocess.STDOUT)
        logger.info("✅ NVIDIA GPU detected.")
        return "gpu"
    except: pass
    if shutil.which("realesrgan-ncnn-vulkan"):
        logger.info("✅ Vulkan GPU detected.")
        return "gpu"
    logger.info("⚠️ No GPU detected. Defaulting to CPU.")
    return "cpu"

def _get_compute_mode():
    mode = os.getenv("COMPUTE_MODE")
    if mode in ["cpu", "gpu", "hybrid"]: return mode
    
    print("\n" + "="*50)
    print("🖥️  COMPUTE MODE SELECTION")
    print("="*50)
    print("1) CPU only")
    print("2) GPU only")
    print("3) Hybrid")
    print("\nAuto-selecting in 10 seconds...")
    
    import msvcrt
    start = time.time()
    inp = ""
    print("Enter choice (1/2/3): ", end="", flush=True)
    
    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char == '\r' or char == '\n':
                print()
                break
            inp += char
        if time.time() - start > 10:
            print("\n⏳ Timeout. Auto-detecting...")
            inp = ""
            break
        time.sleep(0.1)
        
    c = inp.strip()
    if c == "1": mode = "cpu"
    elif c == "2": mode = "gpu"
    elif c == "3": mode = "hybrid"
    else: mode = _detect_hardware()
    
    with open(".env", "a") as f:
        f.write(f"\nCOMPUTE_MODE={mode}\n")
    return mode

# ==================== COMPILATION LOGIC ====================

async def maybe_compile_and_upload(update: Update):
    count = total_uploads()
    n = COMPILATION_BATCH_SIZE
    if n <= 0 or count == 0 or count % n != 0:
        return

    await safe_reply(update, f"⏳ Creating compilation of last {n} shorts...📦")
    files = last_n_filepaths(n)
    if len(files) < n:
        await safe_reply(update, "⚠️ Not enough local files to compile. Skipping.")
        return

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = f"compilation_{n}_{stamp}.mp4"
    await safe_reply(update, f"🔨 Merging {len(files)} videos now...🛸")

    try:
        await safe_reply(update, "✨ Running full AI pipeline for batch compilation…")

        # --- Single Stage: Batch Compile with Transitions ---
        # This replaces the old 2-stage process (raw merge -> enhance)
        # Now we normalize -> transition -> merge -> remix -> assemble in one go
        
        output_filename = f"compilation_{n}_{stamp}.mp4"
        
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            files,
            output_filename
        )
        
        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Failed to create compilation.")
            return

        # Check if we should send to YouTube or Telegram
        send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
        
        if not send_to_youtube:
            await safe_reply(update, "📤 Sending compilation for testing...")
            if os.path.getsize(merged) < 50 * 1024 * 1024:
                await safe_video_reply(update, merged)
            else:
                await safe_reply(update, "⚠️ Compilation too large for Telegram.")
            return

        comp_title = f"🎬 {n} Videos Compilation #{count // n}"  # Changed from "Shorts" to "Videos"
        
        # Use compilation hashtags WITHOUT #Shorts to ensure it's uploaded as regular video
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").replace("#Shorts", "").replace("#shorts", "").strip()
        
        comp_link = await uploader.upload_to_youtube(merged, comp_hashtags, comp_title)

        if comp_link:
            await safe_reply(update, f"🎉 Compilation uploaded!\n🔗 {comp_link}")
            log_video(merged, comp_link, comp_title)
        else:
            await safe_reply(update, "❌ Failed to upload compilation.")

    except Exception as e:
        logger.exception("Compilation/upload failed: %s", e)
        await safe_reply(update, f"❌ Compilation failed: {e}")

# ==================== HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "❓ Please send an Instagram reel or YouTube link to begin.")

async def getbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, f"Current compilation batch size: {COMPILATION_BATCH_SIZE}")

async def setbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global COMPILATION_BATCH_SIZE
    try:
        if not context.args:
            await safe_reply(update, "Usage: /setbatch <number>")
            return
        n = int(context.args[0])
        if n <= 0:
            await safe_reply(update, "Please provide a positive integer.")
            return
        COMPILATION_BATCH_SIZE = n
        await safe_reply(update, f"✅ Compilation batch size set to {n}.")
    except Exception:
        await safe_reply(update, "Usage: /setbatch <number>")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    load_dotenv(override=True)
    send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() == "on"
    
    text = update.message.text.strip()
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    state = session.get('state')
    
    # Case 1: Link
    if _validate_url(text):
        hashtags = _get_hashtags(text)
        user_sessions[user_id] = {'state': 'WAITING_FOR_TITLE', 'url': text, 'hashtags': hashtags}
        await safe_reply(update, "✅ Got the link!")
        await safe_reply(update, f"📌 Hashtags:\n{hashtags}")
        await safe_reply(update, "✏️ Now send the title.")
        return

    # Case 2: Title
    if state == 'WAITING_FOR_TITLE':
        url = session.get('url')
        hashtags = session.get('hashtags')
        title = _sanitize_title(text)
        
        if not title:
            await safe_reply(update, "❌ Invalid title.")
            return
            
        user_sessions.pop(user_id, None)
        
        try:
            await safe_reply(update, "📥 Downloading...")
            video_path = await asyncio.to_thread(downloader.download_video, url, custom_title=title)
            
            if not video_path or not os.path.exists(video_path):
                await safe_reply(update, "❌ Download failed.")
                return

            await safe_reply(update, "🎧 Processing audio...")
            await safe_reply(update, "🧑‍🎨 Enhancing...")
            await safe_reply(update, "🎬 Finalizing output (Transitions Engine)...")
            
            final_path = await asyncio.to_thread(compile_with_transitions, Path(video_path), title)
            
            if not final_path or not os.path.exists(final_path):
                await safe_reply(update, "❌ Processing failed.")
                return
                
            final_str = str(final_path)

            if send_to_youtube:
                await safe_reply(update, "📤 Uploading to YouTube...")
                link = await uploader.upload_to_youtube(final_str, title=title, hashtags=hashtags)
                if link:
                    await safe_reply(update, f"✅ Uploaded: {link}")
                    log_video(final_str, link, title)
                    await maybe_compile_and_upload(update)
                else:
                    await safe_reply(update, "❌ Upload failed. Sending file...")
                    await safe_video_reply(update, final_str)
            else:
                await safe_reply(update, "📤 Sending the video here for testing...")
                if os.path.getsize(final_str) < 50 * 1024 * 1024:
                    await safe_video_reply(update, final_str, caption=f"✨ {title}")
                else:
                    await safe_reply(update, "✅ Video processed! (Too large to send)")
                
                log_video(final_str, "local_test", title)
                await maybe_compile_and_upload(update)

        except Exception as e:
            logger.error(f"Error: {e}")
            await safe_reply(update, "❌ Error occurred.")
        return

    await safe_reply(update, "❓ Please send an Instagram reel or YouTube link to begin.")

def _bootstrap():
    _get_compute_mode()
    if not shutil.which("ffmpeg"):
        logger.error("❌ FFmpeg not found.")
        sys.exit(1)

if __name__ == '__main__':
    if not TOKEN:
        sys.exit(1)
    _bootstrap()
    app = ApplicationBuilder().token(TOKEN).read_timeout(600).write_timeout(600).connect_timeout(60).pool_timeout(60).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("getbatch", getbatch))
    app.add_handler(CommandHandler("setbatch", setbatch))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    logger.info("🤖 Bot is running...")
    app.run_polling()
