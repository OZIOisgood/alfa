---
sidebar_position: 2
---

# Installation

Get ALFA up and running on your machine in minutes!

## Prerequisites

Before installing ALFA, make sure you have:

- **Python 3.13+** - [Download Python](https://www.python.org/downloads/)
- **UV Package Manager** - [Install UV](https://docs.astral.sh/uv/)
- **Docker** (optional but recommended) - [Install Docker](https://www.docker.com/get-started)
- **FFmpeg** - [Download FFmpeg](https://ffmpeg.org/download.html)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/OZIOisgood/alfa.git
cd alfa
```

### 2. Install Python Dependencies

Using UV (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Credentials

#### Google Cloud Credentials

1. Create a Google Cloud project
2. Enable the following APIs:
   - Text-to-Speech API
   - Vertex AI API
3. Create a service account
4. Download the JSON key file
5. Place it in `.credentials/alfa_gcp_sa.json`

#### API Keys

Create a `.env` file in the project root:

```bash
# Google Cloud
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1

# OpenRouter (optional)
OPENROUTER_API_KEY=your-openrouter-key
```

### 4. Build Docker Image (Recommended)

Docker provides isolated rendering with all LaTeX packages:

```bash
docker build -t alfa-manim:latest .
```

### 5. Verify Installation

Test the Gradio app:

```bash
uv run python run-gradio.py
```

Visit `http://127.0.0.1:7865` - you should see the ALFA interface!

## Platform-Specific Notes

### Windows

- Install FFmpeg via [Chocolatey](https://chocolatey.org/):
  ```powershell
  choco install ffmpeg
  ```
- Use PowerShell or Windows Terminal

### macOS

- Install FFmpeg via [Homebrew](https://brew.sh/):
  ```bash
  brew install ffmpeg
  ```

### Linux

- Install FFmpeg:
  ```bash
  sudo apt-get install ffmpeg  # Ubuntu/Debian
  sudo yum install ffmpeg      # CentOS/RHEL
  ```

## Troubleshooting

### Docker Issues

If Docker build fails:

```bash
# Clean and rebuild
docker system prune -a
docker build --no-cache -t alfa-manim:latest .
```

### LaTeX Errors

If you see LaTeX errors without Docker:

1. Install MiKTeX (Windows) or TeX Live (Linux/Mac)
2. Or use Docker for guaranteed LaTeX support

### Module Not Found

If you see import errors:

```bash
# Reinstall dependencies
uv sync --reinstall
```

### Port Already in Use

If port 7865 is taken:

Edit `apps/gradio/main.py` and change:

```python
app.launch(server_port=7865)  # Change to another port
```

## Updating ALFA

Pull the latest changes:

```bash
git pull origin main
uv sync  # Update dependencies
docker build -t alfa-manim:latest .  # Rebuild Docker image
```

## Next Steps

- 🚀 [Quick Start Guide](./quick-start) - Generate your first video
- ⚙️ [Configuration](./configuration) - Customize ALFA settings
- 📚 [Examples](./examples) - See sample problems

---

Need help? [Open an issue](https://github.com/OZIOisgood/alfa/issues) on GitHub!
