<div align="center">
	<img src="assets/alfa.png" alt="ALFA Logo" height="50%" />
</div>

## What's next

* 📹 Publish demo videos and examples
* 🌐 Deploy web version

# ALFA — Short Math Explainer Generator

> **TL;DR:** ALFA is an app that turns any math problem you paste in into a **30–180s** video explanation with voice-over narration.

## What it does

* You paste a math problem.
* ALFA creates a short, clear, narrated video that shows the **steps** and the **why**.

## Who it’s for

Students, self‑learners, and busy folks who want fast, visual clarity instead of long lectures.

## Why it’s useful

* **One problem → one concise video**
* **Visual-first** explanations that highlight the key idea
* **Plain language** and friendly pacing

## How it works (simple)

1. **Input**: Paste your problem + choose quality & voice-over options
2. **Generate**: AI creates script, animations, and voice-over
3. **Process**: Videos are rendered and combined automatically
4. **Output**: A complete video ready to watch or download

> Built with **Gradio** (UI), **GPT-4** (script generation), **Manim** (animations), **Google TTS** (voice-over), and **FFmpeg** (video processing).

## Quick Start

```bash
# Install dependencies
uv sync

# Start the app
uv run python main.py
```

Visit **http://127.0.0.1:7865** and try it out!

📖 See [QUICK_START.md](QUICK_START.md) for detailed usage guide.

## Status

* ✅ Core features complete: video generation, audio sync, quality selection
* ✅ Docker support for isolated Manim rendering
* ✅ Comprehensive LaTeX package support
* 🚧 Testing and optimization ongoing
* 🎬 Sample videos: Coming soon

## What’s next

* Publish a minimal demo page and 3 example clips
* Add a “Try your own problem” input

## Want to help?

Open an issue with a problem you’d like explained in under two minutes.

---

**ALFA** — *Paste a problem. Get a tiny, tidy video.*
