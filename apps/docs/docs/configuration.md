---
sidebar_position: 4
---

# Configuration

Learn how to configure ALFA for your specific needs.

## Environment Variables

ALFA uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Google Cloud Settings

```bash
# Your Google Cloud project ID
GCP_PROJECT_ID=your-project-id

# GCP region
GCP_LOCATION=us-central1

# Path to service account JSON
GOOGLE_APPLICATION_CREDENTIALS=.credentials/alfa_gcp_sa.json
```

**Setup Instructions:**

1. Create a Google Cloud project
2. Enable APIs:
   - Text-to-Speech API
   - Vertex AI API
3. Create service account with appropriate permissions
4. Download JSON key file
5. Place in `.credentials/alfa_gcp_sa.json`

### OpenRouter Settings (Optional)

For using GPT-4, Claude, and other models:

```bash
OPENROUTER_API_KEY=your-key-here
```

Get your key from [OpenRouter](https://openrouter.ai/keys).

### Application Settings

```bash
# Gradio server
GRADIO_SERVER_PORT=7865
GRADIO_SERVER_NAME=127.0.0.1

# Defaults
DEFAULT_QUALITY=m
DEFAULT_LLM_MODEL=gemini-2.5-flash
DEFAULT_TTS_MODEL=gemini-2.5-flash-tts

# Docker
USE_DOCKER=true
DOCKER_IMAGE=alfa-manim:latest
```

## Model Configuration

### LLM Models

Choose the model that generates scripts and animations:

| Model | Provider | Speed | Quality | Cost |
|-------|----------|-------|---------|------|
| gemini-2.5-flash-lite | Vertex AI | ⚡⚡⚡ | ⭐⭐⭐ | 💰 |
| gemini-2.5-flash | Vertex AI | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 |
| gemini-2.5-pro | Vertex AI | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 |
| gpt-4o | OpenRouter | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰💰 |
| claude-3.5-sonnet | OpenRouter | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰💰💰 |

**Recommendation:** Start with `gemini-2.5-flash` for best balance.

### TTS Models

Choose voice-over quality:

| Model | Speed | Quality | Naturalness |
|-------|-------|---------|-------------|
| gemini-2.5-flash-tts | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| gemini-2.5-pro-tts | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Quality Settings

### Video Quality

Configure in the UI or set default in `.env`:

```bash
# Options: l (480p), m (720p), h (1080p), k (4K)
DEFAULT_QUALITY=m
```

**Quality Comparison:**

| Setting | Resolution | Speed | File Size | Use Case |
|---------|-----------|-------|-----------|----------|
| Low (l) | 480p | ⚡⚡⚡ | ~10 MB | Testing, quick previews |
| Medium (m) | 720p | ⚡⚡ | ~30 MB | General use, sharing |
| High (h) | 1080p | ⚡ | ~150 MB | Production, presentations |
| 4K (k) | 2160p | 🐌 | ~500 MB | Maximum quality, archival |

### Voice-Over Toggle

Enable/disable in UI:

```python
generate_voiceover=True  # Enable narration
generate_voiceover=False # Silent video
```

**Benefits of enabling voice-over:**
- ✅ Complete learning experience
- ✅ Better engagement
- ✅ Accessibility
- ✅ Professional output

**When to disable:**
- Testing/development
- Need faster generation
- Planning to add custom audio
- Budget constraints

## Docker Configuration

### Building the Image

```bash
# Build with default settings
docker build -t alfa-manim:latest .

# Build with no cache
docker build --no-cache -t alfa-manim:latest .
```

### Custom Docker Image

Edit `Dockerfile` to add packages or change base image.

### Docker Resources

Allocate more resources in Docker Desktop:
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ recommended
- **Disk**: 10GB+ free space

## Advanced Settings

### Timing Configuration

Adjust gaps between scenes (in UI):

```python
gap_keyframe_seconds=0.5  # Gap between keyframes
gap_section_seconds=1.0   # Gap between sections
```

### Section Limiting

Limit number of sections for testing:

```python
sections_limit=3  # Generate only first 3 sections
```

Useful for:
- Quick tests
- Debugging prompts
- Partial generation

### Output Directory

Default: `output/{request_id}/`

Structure:
```
output/
└── {request_id}/
    ├── final_video.mp4        # ⭐ Final output
    ├── scenario.json          # Generated script
    ├── audio/                 # Voice-over files
    ├── manim_scenes/          # Python scripts
    ├── videos/                # Rendered scenes
    └── videos_with_audio/     # Combined clips
```

## Subject-Specific Configuration

### Mathematics

Best settings:
- Model: `gemini-2.5-flash` or `gemini-2.5-pro`
- Quality: Medium or High
- Voice-over: Enabled

### Chemistry

Best settings:
- Model: `gemini-2.5-pro` (better molecular structures)
- Quality: High (detail important)
- Voice-over: Enabled

### Physics

Best settings:
- Model: `gemini-2.5-flash` or `gpt-4o`
- Quality: Medium or High
- Voice-over: Enabled

### Computer Science

Best settings:
- Model: `gemini-2.5-pro` or `claude-3.5-sonnet`
- Quality: Medium (faster iteration)
- Voice-over: Enabled

## Performance Tuning

### Speed Optimization

For fastest generation:
```python
quality="l"                      # Low quality
generate_voiceover=False         # No audio
sections_limit=3                 # Fewer sections
llm_model="gemini-2.5-flash-lite"  # Fastest model
```

Expected time: **2-3 minutes**

### Quality Optimization

For best output:
```python
quality="h"                      # High quality
generate_voiceover=True          # With audio
sections_limit=10                # Full sections
llm_model="gemini-2.5-pro"      # Best model
tts_model="gemini-2.5-pro-tts"  # Best voice
```

Expected time: **20-30 minutes**

## Troubleshooting

### Slow Generation

- Lower quality setting
- Reduce sections_limit
- Use faster LLM model
- Ensure Docker has enough resources

### API Errors

- Verify credentials in `.env`
- Check API quotas in Google Cloud Console
- Ensure APIs are enabled
- Check OpenRouter balance (if using)

### Memory Issues

- Reduce video quality
- Close other applications
- Increase Docker memory allocation
- Use smaller problems

## Best Practices

### For Development

```bash
DEFAULT_QUALITY=l
USE_DOCKER=true
GRADIO_SERVER_PORT=7865
```

### For Production

```bash
DEFAULT_QUALITY=h
USE_DOCKER=true
DEFAULT_LLM_MODEL=gemini-2.5-pro
DEFAULT_TTS_MODEL=gemini-2.5-pro-tts
```

### For Cost Optimization

- Use Vertex AI models (cheaper than OpenRouter)
- Start with Low quality for testing
- Disable voice-over during development
- Limit sections for quick iterations

---

## Next Steps

- � [Quick Start Guide](./quick-start) - Generate your first video
- 📖 [Introduction](./intro) - Learn more about ALFA
- 💬 [GitHub Repository](https://github.com/OZIOisgood/alfa) - Report issues or contribute

Need help? [Open an issue](https://github.com/OZIOisgood/alfa/issues)!
