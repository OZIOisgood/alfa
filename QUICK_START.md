# Quick Start Guide

## 🚀 Running the Application

```bash
uv run python main.py
```

The app will start at: **http://127.0.0.1:7865**

---

## 🎯 Using the Interface

### Input Section (Left Column)

1. **Math Problem** - Enter your math problem
   ```
   Example: "Simplify: (2x + 3)(x - 4)"
   ```

2. **Video Quality** - Choose resolution
   - 🟢 **Low (480p)** - Fast, ~30-60 sec/scene
   - 🔵 **Medium (720p)** - Default, ~1-2 min/scene ⭐
   - 🟡 **High (1080p)** - Quality, ~2-4 min/scene
   - 🔴 **4K (2160p)** - Pro, ~5-10 min/scene

3. **Generate Voice-over** - Toggle audio
   - ✅ **Enabled** - Full experience with narration
   - ❌ **Disabled** - Silent video (faster)

4. **Gap Settings** - Fine-tune timing
   - Keyframe gap: 0-3 seconds (default: 0.5s)
   - Section gap: 0-5 seconds (default: 1.0s)

5. **Click "Generate Video"** 🎬

### Output Section (Right Column)

#### 1. Final Video Player
- Appears when generation is complete
- Play/pause controls
- Download button

#### 2. Video Scenario
- JSON structure of your video
- Sections and keyframes
- Animation prompts
- Voice-over scripts

#### 3. Generation Status
- Real-time progress updates
- Success/error indicators
- Output file locations

#### 4. Audio Preview (if enabled)
- Combined audio track
- Useful for reviewing narration

---

## 📋 Typical Workflow

### Scenario 1: Full Production Video
```
Problem: "Solve the quadratic equation: x² - 5x + 6 = 0"
Quality: High (1080p)
Voice-over: Enabled ✅
Time: ~10-15 minutes
Result: Professional video with narration
```

### Scenario 2: Quick Preview
```
Problem: "Find derivative of 3x²"
Quality: Low (480p)
Voice-over: Disabled ❌
Time: ~2-3 minutes
Result: Fast silent animation
```

### Scenario 3: Testing
```
Problem: "What is 2 + 2?"
Quality: Low (480p)
Voice-over: Enabled ✅
Time: ~3-5 minutes
Result: Quick test with minimal scenes
```

---

## 📁 Output Files

After generation, find your files in:
```
output/{request_id}/
├── final_video.mp4           ⭐ YOUR FINAL VIDEO
├── scenario.json             📝 Video structure
├── audio/                    🎙️ Voice-over files
│   └── *.wav
├── manim_scenes/             📜 Python scripts
│   └── *.py
├── videos/                   🎬 Individual animations
│   └── *.mp4
└── videos_with_audio/        🎥 Combined clips
    └── *_with_audio.mp4
```

---

## 🎬 Video Generation Pipeline

```
┌─────────────────────────────────────────────────┐
│ 1. Generate Scenario (GPT-3.5)                  │
│    • Parse math problem                         │
│    • Create sections & keyframes                │
│    • Generate animation prompts                 │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 2. Generate Manim Scripts (GPT-4)               │
│    • One script per keyframe                    │
│    • LaTeX math expressions                     │
│    • Animations & transitions                   │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 3. Render Animations (Docker/Local)             │
│    • Manim Community Edition                    │
│    • Selected quality (480p-4K)                 │
│    • Individual MP4 files                       │
└──────────────────┬──────────────────────────────┘
                   ↓
           ┌───────┴───────┐
           ↓               ↓
┌──────────────────┐  ┌──────────────────┐
│ 4a. Generate     │  │ 4b. Skip Audio   │
│     Voice-overs  │  │     (if disabled)│
│ • Google TTS     │  │                  │
│ • WAV files      │  │                  │
└────────┬─────────┘  └────────┬─────────┘
         ↓                     ↓
┌──────────────────┐  ┌──────────────────┐
│ 5a. Combine      │  │ 5b. Use Videos   │
│     Video+Audio  │  │     As-Is        │
│ • Auto timing    │  │                  │
│ • FFmpeg mix     │  │                  │
└────────┬─────────┘  └────────┬─────────┘
         └───────┬───────┬──────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ 6. Concatenate All Videos (FFmpeg)              │
│    • Seamless transitions                       │
│    • Single final_video.mp4                     │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 7. Display in Gradio                            │
│    • Video player                               │
│    • Download option                            │
│    • Status & logs                              │
└─────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Required Environment Variables
```bash
# .env file
OPENROUTER_API_KEY=your_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
```

### Docker Setup (Optional but Recommended)
```bash
# Build Docker image for Manim
docker build -t alfa-manim:latest .

# Test the image
docker run --rm alfa-manim:latest --version
```

---

## 🐛 Troubleshooting

### Video Generation Fails
- ✅ Check Docker is running
- ✅ Verify `alfa-manim:latest` image exists
- ✅ Check disk space (videos can be large)

### Audio Generation Fails
- ✅ Verify `GOOGLE_APPLICATION_CREDENTIALS` in `.env`
- ✅ Check Google Cloud TTS API is enabled
- ✅ Verify service account has permissions

### Slow Generation
- ✅ Use lower quality (480p/720p)
- ✅ Disable voice-over for testing
- ✅ Simplify math problem (fewer steps)

### FFmpeg Not Found
```bash
# Windows (via Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

---

## 💡 Tips & Best Practices

### For Best Results:
1. **Start simple** - Test with basic problems first
2. **Use Medium quality** - Good balance of speed/quality
3. **Check scenario** - Review JSON before full render
4. **Save outputs** - Copy `request_id` for later reference

### For Faster Iteration:
1. **Disable voice-over** during testing
2. **Use Low quality** for previews
3. **Simplify problems** to fewer steps
4. **Keep animations short** (aim for 2-4 scenes)

### For Production:
1. **Enable voice-over** for narration
2. **Use High/4K quality** for final output
3. **Test with Low first** to verify structure
4. **Allow time** - 4K can take 30+ minutes

---

## 📊 Example Outputs

### Simple Problem (Fast)
```
Problem: "What is 5 × 3?"
Scenes: 2-3
Duration: ~10-15 seconds
Render Time: 2-3 minutes (Low)
File Size: ~5 MB
```

### Medium Problem (Normal)
```
Problem: "Solve: 2x + 5 = 15"
Scenes: 4-5
Duration: ~30-45 seconds
Render Time: 8-12 minutes (Medium)
File Size: ~30 MB
```

### Complex Problem (Long)
```
Problem: "Derive quadratic formula"
Scenes: 8-10
Duration: ~90-120 seconds
Render Time: 30-40 minutes (High)
File Size: ~150 MB
```

---

## 🎓 Learning Resources

### Understanding Manim
- [Manim Community Docs](https://docs.manim.community/)
- [Manim Tutorial](https://www.youtube.com/watch?v=rUsUrbWb2D4)

### LaTeX Math
- See `prompts/manim_scene.txt` for available packages
- [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)

### FFmpeg
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Video Filtering](https://ffmpeg.org/ffmpeg-filters.html)

---

## 📞 Support

Issues? Check:
1. **Generation Status** - Error messages in UI
2. **Terminal Output** - Detailed logs
3. **Output Files** - Check if partial files exist
4. **Docker Logs** - If using Docker rendering

---

**Ready to create amazing math videos? Start the app and try it out! 🚀**
