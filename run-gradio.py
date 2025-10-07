#!/usr/bin/env python3
"""
Launch script for ALFA Gradio application.
This allows running the app from the root directory.
"""
import sys
from pathlib import Path

# Add the gradio app directory to Python path
gradio_app_dir = Path(__file__).parent / "apps" / "gradio"
sys.path.insert(0, str(gradio_app_dir))

# Import and run the main app
from main import main

if __name__ == "__main__":
    print("🚀 Starting ALFA Gradio Application...")
    print(f"📁 App directory: {gradio_app_dir}")
    main()
