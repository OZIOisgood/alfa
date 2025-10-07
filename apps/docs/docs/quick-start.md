---
sidebar_position: 3
---

# Quick Start

Generate your first video in under 5 minutes!

## Step 1: Launch the App

From the project root:

```bash
uv run python run-gradio.py
```

The app will start at **http://127.0.0.1:7865**

## Step 2: Select Subject

Choose from the dropdown:
- 📐 Mathematics
- 🧪 Chemistry  
- ⚡ Physics
- 💻 Computer Science

## Step 3: Enter Problem

Type or paste your problem. Examples:

**Math:**
```
Find the area of a circle with radius 5 cm
```

**Chemistry:**
```
Balance the equation: H₂ + O₂ → H₂O
```

**Physics:**
```
A car accelerates from 0 to 20 m/s in 5 seconds. Find acceleration.
```

**Computer Science:**
```
Explain binary search on array [3, 7, 12, 18, 25]
```

## Step 4: Configure Settings

### Quality (Video Resolution)
- **Low (480p)** - Fast, good for testing (~2-3 min)
- **Medium (720p)** - Recommended balance (~ 2-4 min)
- **High (1080p)** - Production quality (~15-25 min)
- **4K (2160p)** - Maximum quality (~30-60 min)

### Voice-over
- ✅ **Enabled** - Adds AI narration (recommended)
- ❌ **Disabled** - Silent video (faster)

### LLM Model
- **Gemini 2.5 Flash** - Fast, good quality (recommended)
- **Gemini 2.5 Pro** - Best quality, slower
- **GPT-4o** - Alternative LLM (requires OpenRouter)

### TTS Model (if voice-over enabled)
- **Gemini 2.5 Flash TTS** - Fast, natural voice
- **Gemini 2.5 Pro TTS** - Highest quality voice

## Step 5: Generate!

Click **"Generate Video"** and wait for:

1. ✅ Scenario generation
2. ✅ Script creation  
3. ✅ Scene rendering
4. ✅ Voice-over generation
5. ✅ Final video compilation

Progress shows in real-time!

## Step 6: Download & Share

Once complete:
- 📺 Watch the video in the player
- 💾 Download the MP4 file
- 📁 Find it in `output/{request_id}/final_video.mp4`

## Example Workflow

### Quick Test (Low Quality, No Voice)
```
Subject: Mathematics
Problem: "Solve: 2x + 5 = 15"
Quality: Low (480p)
Voice-over: Disabled
Time: ~2 minutes
```

### Production Video (High Quality + Voice)
```
Subject: Chemistry
Problem: "Balance: CH₄ + O₂ → CO₂ + H₂O"
Quality: High (1080p)
Voice-over: Enabled
LLM: Gemini 2.5 Pro
TTS: Gemini 2.5 Pro TTS
Time: ~20 minutes
```

## Tips for Best Results

### ✅ Do:
- Keep problems focused (one concept)
- Use clear, specific wording
- Start with Low quality for testing
- Enable voice-over for complete experience

### ❌ Avoid:
- Multiple unrelated concepts
- Vague or ambiguous problems
- Very complex proofs (break into parts)
- Starting with 4K (test first!)

## Troubleshooting

### No video generated?
- Check terminal for errors
- Verify Docker is running (if using Docker)
- Try simpler problem first

### Audio issues?
- Verify Google Cloud credentials
- Check .env file configuration
- Try VLC media player (better codec support)

### Slow rendering?
- Lower quality setting
- Disable voice-over temporarily
- Check Docker resources

## Next Steps

- 📚 Explore [Examples](./examples) for more ideas
- ⚙️ Learn about [Configuration](./configuration) options
- 🎨 Read subject-specific guides:
  - [Mathematics](./subjects/math)
  - [Chemistry](./subjects/chemistry)
  - [Physics](./subjects/physics)
  - [Computer Science](./subjects/cs)

---

**Ready to create amazing educational content!** 🚀
