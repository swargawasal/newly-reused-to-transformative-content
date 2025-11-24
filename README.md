# 🎬 YouTube Automation Bot - AI Video Enhancement

**Transform reused content into viral-ready videos with AI enhancement, smart editing, and GPU acceleration.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## ✨ Features

### 🚀 **Performance Modes**

- **FAST_MODE**: 10x faster processing with FFmpeg (30s vs 51 hours!)
- **AI Enhancement**: GPU-accelerated upscaling with Real-ESRGAN + GFPGAN
- **Smart Hardware Detection**: Auto-detects GPU/CPU and optimizes accordingly

### 🎨 **Transformative Content**

- ✅ AI-powered video upscaling (2x/4x)
- ✅ Face enhancement (GFPGAN)
- ✅ Dynamic text overlays
- ✅ Cinematic color grading
- ✅ Speed ramping & zoom effects
- ✅ Professional audio remixing
- ✅ Smooth transitions

### 🤖 **Telegram Bot Integration**

- Send videos via Telegram
- Automatic processing
- Batch compilation support
- YouTube upload ready

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended for GPU)**

1. **Clone the repository:**

   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   %cd YOUR_REPO
   ```

2. **Create `.env` file:**

   ```python
   %%writefile .env
   TELEGRAM_BOT_TOKEN=your_token_here
   GEMINI_API_KEY=your_key_here
   FAST_MODE=no
   ENHANCEMENT_LEVEL=high
   ```

3. **Run installation:**

   ```python
   !python install_colab.py
   ```

4. **Start the bot:**
   ```python
   !python main.py
   ```

📖 **Full Colab Guide:** See [`COLAB_SETUP.md`](COLAB_SETUP.md)

---

### **Option 2: Local Installation (Windows/Linux)**

1. **Clone and setup:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `.env`:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run:**
   ```bash
   python main.py
   ```

---

## ⚙️ Configuration

### **Performance Modes**

| Mode                       | Speed      | Quality    | VRAM    | Best For               |
| -------------------------- | ---------- | ---------- | ------- | ---------------------- |
| `FAST_MODE=yes`            | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | 0 GB    | Quick tests, CPU-only  |
| `ENHANCEMENT_LEVEL=basic`  | ⚡⚡⚡⚡   | ⭐⭐⭐     | 0 GB    | Fast processing        |
| `ENHANCEMENT_LEVEL=medium` | ⚡⚡⚡     | ⭐⭐⭐⭐   | 4-6 GB  | Balanced (Recommended) |
| `ENHANCEMENT_LEVEL=high`   | ⚡⚡       | ⭐⭐⭐⭐⭐ | 6-8 GB  | Best quality           |
| `ENHANCEMENT_LEVEL=ultra`  | ⚡         | ⭐⭐⭐⭐⭐ | 8-12 GB | Maximum quality        |

### **Key Environment Variables**

```ini
# Core
TELEGRAM_BOT_TOKEN=your_token        # From @BotFather
GEMINI_API_KEY=your_key              # From Google AI Studio

# Performance
FAST_MODE=no                         # yes = FFmpeg only, no = AI
ENHANCEMENT_LEVEL=medium             # basic/medium/high/ultra
COMPUTE_MODE=auto                    # auto/cpu/gpu

# Transformative Features
ADD_TEXT_OVERLAY=yes                 # Add viral text
ADD_COLOR_GRADING=yes                # Cinematic look
ADD_SPEED_RAMPING=yes                # Dynamic speed
FORCE_AUDIO_REMIX=yes                # Professional audio
```

---

## 🎯 Hardware Requirements

### **Minimum (CPU Mode)**

- CPU: Any modern processor
- RAM: 4 GB
- Storage: 2 GB
- Mode: `FAST_MODE=yes`

### **Recommended (GPU Mode)**

- GPU: NVIDIA GTX 1060 or better
- VRAM: 6 GB+
- RAM: 8 GB
- CUDA: 11.8+
- Mode: `ENHANCEMENT_LEVEL=medium`

### **Optimal (Colab T4)**

- GPU: Tesla T4 (15 GB VRAM)
- Mode: `ENHANCEMENT_LEVEL=high` or `ultra`
- Processing: ~30s per video

---

## 📊 Smart Startup System

The bot automatically:

1. **Detects Hardware** (GPU/CPU/VRAM) before loading libraries
2. **Creates `.env`** template if missing
3. **Installs Dependencies** based on mode:
   - `FAST_MODE=yes`: Lightweight packages only
   - `FAST_MODE=no`: Full AI stack + models
4. **Optimizes Configuration** for your hardware

### Example Output:

```
🔧 Starting Smart Startup Checks...
🎮 GPU Detected: Tesla T4 (15.0 GB VRAM)
✅ PyTorch CUDA Available: True
⚡ FAST_MODE: False (Source: no)
📦 Installing full dependencies (AI mode)...
✅ Startup checks complete.
```

---

## 🎬 Usage

### **Telegram Commands**

- `/start` - Start the bot
- Send a video - Process single video
- Send multiple videos - Create compilation

### **Processing Flow**

1. **Send video** → Bot downloads
2. **AI Enhancement** → Upscale + face enhancement
3. **Transformative Effects** → Text, color, speed, audio
4. **Output** → Receive processed video

---

## 🛠️ Troubleshooting

### **Issue: Bot running on CPU despite having GPU**

**Solution:**

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Issue: "CUDA out of memory"**

**Solution:** Lower enhancement level in `.env`:

```ini
ENHANCEMENT_LEVEL=medium  # or basic
```

### **Issue: scipy import error**

**Solution:** Enable FAST_MODE:

```ini
FAST_MODE=yes
```

### **Issue: Bot not responding**

**Solution:** Check bot token:

```bash
cat .env | grep TELEGRAM_BOT_TOKEN
```

---

## 📁 Project Structure

```
├── main.py                 # Bot entry point
├── compiler.py             # Video processing engine
├── ai_engine.py            # AI enhancement (Real-ESRGAN, GFPGAN)
├── downloader.py           # Video downloader
├── uploader.py             # YouTube uploader
├── audio_processing.py     # Audio remixing
├── tools-install.py        # AI model installer
├── install_colab.py        # Colab setup script
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
├── COLAB_SETUP.md          # Colab guide
└── README.md               # This file
```

---

## 🔒 Security

- **Never commit `.env`** to Git (it's in `.gitignore`)
- **Use Colab Secrets** for sensitive data in Colab
- **Rotate API keys** regularly

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Credits

- **Real-ESRGAN** - AI upscaling
- **GFPGAN** - Face enhancement
- **FFmpeg** - Video processing
- **python-telegram-bot** - Telegram integration
- **Google Gemini** - AI assistance

---

## 🚀 Performance Benchmarks

| Hardware          | Mode        | Video (23s) | Processing Time |
| ----------------- | ----------- | ----------- | --------------- |
| CPU (i7)          | FAST_MODE   | 23s         | ~30s            |
| CPU (i7)          | AI (medium) | 23s         | ~51 hours ❌    |
| GPU (GT 730, 2GB) | AI (medium) | 23s         | ~15 min         |
| GPU (T4, 15GB)    | AI (high)   | 23s         | ~30s ✅         |
| GPU (T4, 15GB)    | AI (ultra)  | 23s         | ~1 min          |

**Recommendation:** Use `FAST_MODE=yes` on CPU, `ENHANCEMENT_LEVEL=high` on T4 GPU.

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/YOUR_REPO/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/YOUR_REPO/discussions)

---

**Made with ❤️ for content creators**
