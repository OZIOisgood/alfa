"""
Test script to verify audio combination with Manim videos works correctly.
"""
import subprocess
from pathlib import Path

# Test the fixed FFmpeg command
video_path = r"C:\Users\p.lobariev\Documents\myProjects\alfa\output\6eb3e6811a32\videos\IntroductionToTheProblemProblemIntroduction.mp4"
audio_path = r"C:\Users\p.lobariev\Documents\myProjects\alfa\output\6eb3e6811a32\audio\introduction_to_the_problem_problem_introduction_1e130ac9.wav"
output_path = r"C:\Users\p.lobariev\Documents\myProjects\alfa\output\6eb3e6811a32\test_combined.mp4"

print("Testing FFmpeg audio combination (no delay)...")
print(f"Video: {video_path}")
print(f"Audio: {audio_path}")
print(f"Output: {output_path}")

# Simple command without delay (like our fixed code)
cmd = [
    'ffmpeg', '-y',
    '-i', video_path,
    '-i', audio_path,
    '-map', '0:v',
    '-map', '1:a',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-shortest',
    output_path
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("\n✅ Success! Audio combination worked.")
    print(f"\nOutput file created: {output_path}")
    
    # Check if file exists and has size
    output = Path(output_path)
    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error: {e}")
    print(f"\nStderr:\n{e.stderr}")
