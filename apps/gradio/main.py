import os
import json
import wave
import subprocess
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from google.cloud import texttospeech
from google.oauth2 import service_account
import numpy as np
import hashlib
import re
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import time
import requests
import warnings
import logging

# Suppress warnings and unnecessary logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)
PCM_CHANNELS = 1
PCM_SAMPLE_RATE = 24000

# Load environment variables
load_dotenv()

def initialize_vertex_ai():
    """
    Initialize Vertex AI with credentials from service account.
    """
    # Navigate from apps/gradio to root .credentials folder
    credentials_path = Path(__file__).parent.parent.parent / ".credentials" / "alfa_gcp_sa.json"
    project_id = os.getenv("GCP_PROJECT_ID", "alfa-473522")  # Default project ID
    location = os.getenv("GCP_LOCATION", "us-central1")  # Default location
    
    try:
        credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
        vertexai.init(project=project_id, location=location, credentials=credentials)
        return True
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return False

def load_prompt_template(template_name):
    """
    Load a prompt template from the prompts directory.
    """
    prompts_dir = Path(__file__).parent / "prompts"
    template_path = prompts_dir / f"{template_name}.txt"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Prompt template '{template_name}' not found in prompts directory."
    except Exception as e:
        return f"Error loading prompt template: {str(e)}"


def format_seconds_to_timestamp(seconds):
    """
    Convert a length in seconds to an MM:SS timestamp string.
    """
    if seconds is None:
        return None

    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def sanitize_class_name(text):
    """Convert a string into a valid Python class name."""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', text.replace(' ', '_').replace('-', '_'))
    # Ensure it starts with a letter or underscore
    if cleaned and not cleaned[0].isalpha() and cleaned[0] != '_':
        cleaned = '_' + cleaned
    # Capitalize first letter of each word
    return ''.join(word.capitalize() for word in cleaned.split('_'))


def generate_manim_scene(section_name, keyframe_name, animation_prompt, voice_over_duration, llm_model="gemini-2.5-flash", previous_scene_code=None):
    """
    Use Vertex AI or OpenRouter to generate a Manim scene Python script for a keyframe.
    
    Args:
        section_name: Name of the section
        keyframe_name: Name of the keyframe
        animation_prompt: Prompt for the animation
        voice_over_duration: Duration of voice-over in seconds
        llm_model: Model to use (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro, or openrouter models)
        previous_scene_code: Code from the previous scene for context continuity
    """
    # Determine if using OpenRouter or Vertex AI
    use_openrouter = llm_model.startswith("openrouter/")
    
    if not use_openrouter:
        # Initialize Vertex AI
        if not initialize_vertex_ai():
            return None, "Error: Failed to initialize Vertex AI"
    
    # Load Manim prompt template
    prompt_template = load_prompt_template("manim_scene")
    if prompt_template.startswith("Error:"):
        return None, prompt_template
    
    # Generate valid class name
    class_name = sanitize_class_name(f"{section_name}_{keyframe_name}")
    if not class_name:
        class_name = "ManimScene"
    
    # Format the prompt
    formatted_prompt = prompt_template.replace("{section_name}", section_name)
    formatted_prompt = formatted_prompt.replace("{keyframe_name}", keyframe_name)
    formatted_prompt = formatted_prompt.replace("{voice_over_duration}", str(voice_over_duration))
    formatted_prompt = formatted_prompt.replace("{animation_prompt}", animation_prompt)
    formatted_prompt = formatted_prompt.replace("{class_name}", class_name)
    
    # Add previous scene context if available
    if previous_scene_code:
        formatted_prompt += f"\n\nPREVIOUS SCENE CODE (for continuity):\n```python\n{previous_scene_code}\n```\n\nUse this as reference for visual consistency, but create a NEW scene with the class name {class_name}."
    
    # Add design consistency constraints
    formatted_prompt += "\n\nDESIGN CONSISTENCY RULES (CRITICAL - MUST FOLLOW):\n"
    formatted_prompt += "1. BACKGROUND: Always use BLACK background (self.camera.background_color = BLACK)\n"
    formatted_prompt += "2. TEXT COLOR: Use WHITE or light colors (YELLOW, CYAN, GREEN) - NEVER dark blue or gray on black background\n"
    formatted_prompt += "3. MAIN OBJECTS: Use consistent colors throughout:\n"
    formatted_prompt += "   - Circles/shapes: WHITE or BLUE outline\n"
    formatted_prompt += "   - Formulas: WHITE or YELLOW text\n"
    formatted_prompt += "   - Highlights: Use YELLOW, GREEN, or RED (never dark colors)\n"
    formatted_prompt += "4. If previous scene had specific colors for objects, MAINTAIN those exact colors\n"
    formatted_prompt += "5. CONTRAST: Ensure high contrast - light objects on dark background\n"
    
    # Add frame boundary constraints
    formatted_prompt += "\nFRAME BOUNDARY CONSTRAINTS:\n"
    formatted_prompt += "1. Keep ALL text and objects within frame boundaries: use config.frame_width and config.frame_height\n"
    formatted_prompt += "2. Recommended safe zone: x between -5 and 5, y between -3 and 3\n"
    formatted_prompt += "3. Use .scale() to ensure objects fit within frame\n"
    formatted_prompt += "4. Use .move_to() with coordinates that stay within boundaries\n"
    formatted_prompt += "5. Return ONLY the Python code, without markdown code blocks or any additional text."
    
    try:
        if use_openrouter:
            # Use OpenRouter API
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return None, "Error: OPENROUTER_API_KEY not found"
            
            # Extract model name (remove "openrouter/" prefix)
            model_name = llm_model.replace("openrouter/", "")
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost:7865",
                "X-Title": "Alfa Video Generator"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 8192
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            code = result['choices'][0]['message']['content']
            
        else:
            # Use Vertex AI Gemini
            model = GenerativeModel(llm_model)
            
            response = model.generate_content(
                formatted_prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                    response_mime_type="text/plain"
                )
            )
            
            code = response.text
        
        # Extract Python code from markdown if present (fallback safety)
        code_match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        
        return code, None
        
    except Exception as e:
        return None, f"Error generating Manim code: {str(e)}"


def pcm_bytes_to_numpy(audio_bytes, sample_width=PCM_SAMPLE_WIDTH):
    """Convert PCM bytes into a numpy array."""
    if audio_bytes is None:
        return None

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 1:
        dtype = np.int8
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("Unsupported sample width")

    return np.frombuffer(audio_bytes, dtype=dtype)


def generate_silence(duration_seconds, sample_rate=PCM_SAMPLE_RATE, sample_width=PCM_SAMPLE_WIDTH):
    """Generate silence of the specified duration as a numpy array."""
    if duration_seconds <= 0:
        return None

    num_samples = int(round(duration_seconds * sample_rate))
    if num_samples <= 0:
        return None

    if sample_width == 2:
        return np.zeros(num_samples, dtype=np.int16)
    elif sample_width == 1:
        return np.zeros(num_samples, dtype=np.int8)
    elif sample_width == 4:
        return np.zeros(num_samples, dtype=np.int32)

    raise ValueError("Unsupported sample width for silence generation")


def apply_fade(samples, sample_rate=PCM_SAMPLE_RATE, fade_duration=0.1):
    """Apply a short fade-in and fade-out to reduce clicks at boundaries.
    
    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        fade_duration: Duration of fade in seconds (default: 0.1s = 100ms to prevent clicking)
    """
    if samples is None or samples.size == 0:
        return samples

    fade_samples = int(round(fade_duration * sample_rate))
    if fade_samples <= 0:
        return samples

    fade_samples = min(fade_samples, samples.size // 2)
    if fade_samples <= 0:
        return samples

    working = samples.astype(np.float32, copy=True)

    fade_in_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    working[:fade_samples] *= fade_in_curve

    fade_out_curve = fade_in_curve[::-1]
    working[-fade_samples:] *= fade_out_curve

    # Clip back to int16 range and cast
    np.clip(working, np.iinfo(np.int16).min, np.iinfo(np.int16).max, out=working)
    return working.astype(np.int16)


def combine_audio_arrays(audio_results, gap_keyframe_seconds, gap_section_seconds, sample_rate=PCM_SAMPLE_RATE):
    """Combine individual PCM arrays with configurable gaps."""
    if not audio_results:
        return None, "No audio results to combine"

    segments = []
    previous_section = None

    for result in audio_results:
        if previous_section is not None:
            gap_seconds = gap_section_seconds if result['section'] != previous_section else gap_keyframe_seconds
            if gap_seconds > 0:
                silence = generate_silence(gap_seconds, sample_rate)
                if silence is not None:
                    segments.append(silence)

        samples = pcm_bytes_to_numpy(result.get('audio_bytes'))
        if samples is None:
            return None, "Missing audio bytes for one or more tracks"

        samples = apply_fade(samples, sample_rate=sample_rate)
        segments.append(samples)
        previous_section = result['section']

    if not segments:
        return None, "No audio segments to combine"

    combined = np.concatenate(segments)
    return combined, None


def numpy_audio_to_gradio_value(samples, sample_rate=PCM_SAMPLE_RATE):
    """Convert a 1-D numpy array into a Gradio-compatible audio tuple."""
    if samples is None:
        return None

    if samples.ndim > 1:
        data = samples
    else:
        data = np.ascontiguousarray(samples)

    return sample_rate, data


def get_video_duration(video_path):
    """Get duration of a video file using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 
            'format=duration', '-of', 
            'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return None


def combine_video_with_audio(video_path, audio_path, output_path, audio_timing='auto'):
    """
    Combine a video file with an audio file.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path for output video
        audio_timing: How to sync audio - 'start', 'center', 'auto', or float (delay in seconds)
    
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        video_duration = get_video_duration(video_path)
        if video_duration is None:
            return False, "Could not determine video duration"
        
        # Get audio duration
        with wave.open(str(audio_path), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            audio_duration = frames / float(rate)
        
        # Determine audio delay
        if audio_timing == 'auto':
            # Auto: If video is much longer than audio, center the audio
            # Otherwise start immediately
            if video_duration > audio_duration * 1.5:
                delay = (video_duration - audio_duration) / 2
            else:
                delay = 0
        elif audio_timing == 'center':
            delay = max(0, (video_duration - audio_duration) / 2)
        elif audio_timing == 'start':
            delay = 0
        elif isinstance(audio_timing, (int, float)):
            delay = max(0, float(audio_timing))
        else:
            delay = 0
        
        # Build ffmpeg command
        # Note: Manim videos have no audio stream, so we just add the audio as a new stream
        # Use video duration as master, loop audio if too short, or trim if too long
        if delay > 0:
            # If there's a delay, use adelay filter
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(video_path),
                '-i', str(audio_path),
                '-filter_complex',
                f'[1:a]adelay={int(delay * 1000)}|{int(delay * 1000)},apad[aout]',  # Add padding at end
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',  # Use video duration as master
                str(output_path)
            ]
        else:
            # No delay, just add audio directly with padding
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(video_path),
                '-i', str(audio_path),
                '-filter_complex',
                '[1:a]apad[aout]',  # Add silent padding if audio is shorter than video
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',  # Use video duration as master
                str(output_path)
            ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, None
        
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"
    except Exception as e:
        return False, str(e)


def concatenate_videos(video_paths, output_path):
    """
    Concatenate multiple video files into one.
    
    Args:
        video_paths: List of paths to video files
        output_path: Path for output video
    
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    if not video_paths:
        return False, "No videos to concatenate"
    
    try:
        # Create a temporary file list for ffmpeg
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            for video_path in video_paths:
                # FFmpeg concat requires absolute paths and proper escaping
                abs_path = Path(video_path).resolve()
                # Escape single quotes and wrap in single quotes
                escaped = str(abs_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            temp_list = f.name
        
        # Concatenate using ffmpeg
        # Explicitly copy both video and audio streams
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list,
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'copy',  # Copy audio stream
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up temp file
        os.unlink(temp_list)
        
        return True, None
        
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"
    except Exception as e:
        return False, str(e)


def render_manim_scene(script_path, class_name, output_dir, quality='m', use_docker=True):
    """
    Render a Manim scene to video using Docker (preferred) or local manim command.
    
    Args:
        script_path: Path to the Python file containing the scene
        class_name: Name of the Scene class to render
        output_dir: Directory where output video should be saved
        quality: Quality setting (l=480p, m=720p, h=1080p, k=4K)
        use_docker: Whether to use Docker for rendering (default: True)
    
    Returns:
        (video_path, error_message) tuple
    """
    try:
        # Prepare output directory
        video_output_dir = Path(output_dir) / "videos"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        quality_flag = f"-q{quality}"
        script_parent = Path(script_path).parent
        script_name = Path(script_path).stem
        quality_map = {'l': '480p15', 'm': '720p30', 'h': '1080p60', 'k': '2160p60'}
        quality_folder = quality_map.get(quality, '720p30')
        
        if use_docker:
            # Use Docker for isolated rendering with LaTeX support
            import os
            
            # Get absolute paths for Docker volume mounting
            project_root = Path(__file__).parent.resolve()
            
            # Handle both relative and absolute paths
            script_path_obj = Path(script_path)
            if not script_path_obj.is_absolute():
                # Relative path - resolve from project root
                script_abs = (project_root / script_path_obj).resolve()
            else:
                script_abs = script_path_obj.resolve()
            
            # Calculate relative path from project root to script
            try:
                script_rel = script_abs.relative_to(project_root)
            except ValueError:
                # Script is outside project root, fallback to local
                return render_manim_scene(script_path, class_name, output_dir, quality, use_docker=False)
            
            # Docker volume mount paths (Windows style -> Linux container paths)
            # Mount project root as /manim in container
            volume_mount = f"{project_root}:/manim"
            
            # Container paths
            container_script_path = f"/manim/{script_rel.as_posix()}"
            container_media_dir = f"/manim/{script_rel.parent.as_posix()}/media"
            
            # Build Docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", volume_mount,
                "-w", "/manim",
                "alfa-manim:latest",
                quality_flag,
                "--media_dir", container_media_dir,
                container_script_path,
                class_name
            ]
            
            # Run Docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace'  # Replace non-UTF8 characters instead of failing
            )
            
            if result.returncode != 0:
                # Safely combine output streams
                error_msg = (result.stderr or "") + (result.stdout or "")
                
                # Check if Docker is available
                if "docker" in error_msg.lower() and ("not found" in error_msg.lower() or "cannot connect" in error_msg.lower()):
                    # Fallback to local rendering
                    print("⚠️ Docker not available, falling back to local manim...")
                    return render_manim_scene(script_path, class_name, output_dir, quality, use_docker=False)
                
                # Check for other errors
                if "SyntaxWarning" in error_msg and "invalid escape sequence" in error_msg:
                    return None, "Generated code has LaTeX escape sequence error. Please regenerate the scene."
                else:
                    return None, f"Docker render failed: {error_msg[:800]}"
        
        else:
            # Local rendering (fallback)
            import os
            import shutil as sh
            env = os.environ.copy()
            
            # Try to locate ffmpeg explicitly and add to PATH if found
            ffmpeg_path = sh.which("ffmpeg")
            if ffmpeg_path:
                ffmpeg_dir = str(Path(ffmpeg_path).parent)
                if ffmpeg_dir not in env.get("PATH", ""):
                    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
            
            # Run manim render command with custom media directory
            result = subprocess.run(
                ["manim", quality_flag, "--media_dir", str(script_parent / "media"), str(script_path), class_name],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(script_parent),
                env=env
            )
            
            if result.returncode != 0:
                error_msg = result.stderr + result.stdout
                
                if "SyntaxWarning" in error_msg and "invalid escape sequence" in error_msg:
                    return None, "Generated code has LaTeX escape sequence error. Please regenerate the scene."
                elif "FileNotFoundError" in error_msg and "latex" in error_msg.lower():
                    return None, "LaTeX not installed. Please install MiKTeX or TeX Live to render text/math in animations."
                elif "FileNotFoundError" in error_msg:
                    return None, f"Missing dependency. Full error: {error_msg[:800]}"
                elif "Couldn't find ffmpeg" in error_msg:
                    return None, "FFmpeg not found in PATH. Please ensure ffmpeg is installed and added to your system PATH."
                else:
                    return None, f"Manim render failed: {error_msg[:800]}"
        
        # Find rendered video (same for both Docker and local)
        media_dir = script_parent / "media" / "videos" / script_name / quality_folder
        
        if not media_dir.exists():
            return None, f"Output directory not found: {media_dir}"
        
        video_files = list(media_dir.glob("*.mp4"))
        if not video_files:
            return None, f"No video file found in {media_dir}"
        
        # Copy to output directory
        source_video = video_files[0]
        dest_video = video_output_dir / f"{class_name}.mp4"
        
        import shutil
        shutil.copy2(source_video, dest_video)
        
        return str(dest_video), None
        
    except subprocess.TimeoutExpired:
        return None, f"Render timeout (>5 minutes) for {class_name}"
    except Exception as e:
        return None, f"Render error: {str(e)}"

def initialize_tts_client():
    """
    Initialize Google Cloud Text-to-Speech client with service account credentials.
    """
    # Navigate from apps/gradio to root .credentials folder
    credentials_path = Path(__file__).parent.parent.parent / ".credentials" / "alfa_gcp_sa.json"
    
    try:
        credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        return client
    except Exception as e:
        print(f"Error initializing TTS client: {e}")
        return None

def generate_audio_from_text(text, filename_prefix, output_dir, client=None, tts_model="gemini-2.5-flash-tts", max_retries=3):
    """
    Generate audio from text using Google Cloud Text-to-Speech API with Gemini models.
    
    Args:
        text: Text to convert to speech
        filename_prefix: Prefix for the output audio file
        output_dir: Directory to save the audio file
        client: TTS client (optional, will initialize if not provided)
        tts_model: TTS model to use (gemini-2.5-flash-tts or gemini-2.5-pro-tts)
        max_retries: Maximum number of retries for quota errors
    """
    if client is None:
        client = initialize_tts_client()
        if client is None:
            return None, "Failed to initialize TTS client", None, None
    
    # Retry loop for quota errors
    for attempt in range(max_retries):
        try:
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Select voice model based on tts_model parameter
            # Map tts_model to voice name and model
            if tts_model == "gemini-2.5-pro-tts":
                # Gemini 2.5 Pro TTS (higher quality)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="Achernar",
                    model_name="gemini-2.5-pro-tts"
                )
            else:  # gemini-2.5-flash-tts (default)
                # Gemini 2.5 Flash TTS (faster, good quality)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="Aoede",
                    model_name="gemini-2.5-flash-tts"
                )
            
            # Configure audio format (16-bit PCM for easy processing)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=0.0,
                sample_rate_hertz=PCM_SAMPLE_RATE
            )
            
            # Generate speech
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            audio_bytes = response.audio_content

            # Apply fade to prevent clicks at start/end
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            faded_audio = apply_fade(audio_array, sample_rate=PCM_SAMPLE_RATE, fade_duration=0.1)
            audio_bytes = faded_audio.tobytes()

            # Determine duration from PCM bytes
            bytes_per_second = PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH
            duration_seconds = len(audio_bytes) / bytes_per_second if audio_bytes else 0

            # Save audio file to request-specific directory
            audio_dir = output_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with hash to avoid conflicts
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"{filename_prefix}_{text_hash}.wav"
            audio_path = audio_dir / filename
            
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(PCM_CHANNELS)
                wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
                wav_file.setframerate(PCM_SAMPLE_RATE)
                wav_file.writeframes(audio_bytes)
            
            return str(audio_path), f"Audio generated successfully: {filename}", duration_seconds, audio_bytes
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a quota error
            if "429" in error_msg or "Quota exceeded" in error_msg:
                if attempt < max_retries - 1:
                    # Wait exponentially longer with each retry
                    wait_time = 3 * (2 ** attempt)  # 3s, 6s, 12s
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    return None, f"Error generating audio (quota exceeded after {max_retries} attempts): {error_msg}", None, None
            else:
                # Non-quota error, don't retry
                return None, f"Error generating audio: {error_msg}", None, None
    
    return None, "Error generating audio: Max retries exceeded", None, None

def parse_scenario_json(scenario_text):
    """
    Parse the scenario JSON and extract all voice-over texts with metadata.
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', scenario_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try to find JSON directly
            json_str = scenario_text
        
        scenario_data = json.loads(json_str)
        voice_texts = []
        
        # Extract voice-over texts from sections and keyframes
        for section_index, section in enumerate(scenario_data.get('sections', [])):
            section_name = section.get('section_name', 'Unknown Section')
            for frame_index, frame in enumerate(section.get('key_frames', [])):
                voice_text = frame.get('voice_over_text', '')
                frame_name = frame.get('name', 'Unknown Frame')
                timestamp = frame.get('estimate_length_timestamp', '00:00')
                
                if voice_text:
                    voice_texts.append({
                        'section': section_name,
                        'frame_name': frame_name,
                        'timestamp': timestamp,
                        'text': voice_text,
                        'filename_prefix': f"{section_name.lower().replace(' ', '_')}_{frame_name.lower().replace(' ', '_')}",
                        'section_index': section_index,
                        'frame_index': frame_index
                    })
        
        return scenario_data, voice_texts, None
        
    except json.JSONDecodeError as e:
        return None, [], f"Error parsing scenario JSON: {str(e)}"
    except Exception as e:
        return None, [], f"Error extracting voice texts: {str(e)}"

def generate_scenario_with_audio_and_manim(
    problem_text, 
    gap_keyframe_seconds=0.5, 
    gap_section_seconds=1.0, 
    generate_manim=True,
    generate_voiceover=True,
    quality='m',
    llm_model="gemini-2.5-flash",
    tts_model="gemini-2.5-flash-tts",
    sections_limit=10,
    subject="math"
):
    """
    Generate a video scenario, create audio tracks, and generate Manim animation scripts.
    
    Args:
        problem_text: The problem or concept to explain
        gap_keyframe_seconds: Gap between keyframes in audio
        gap_section_seconds: Gap between sections in audio
        generate_manim: Whether to generate Manim animations
        generate_voiceover: Whether to generate voice-over audio
        quality: Video quality ('l'=480p, 'm'=720p, 'h'=1080p, 'k'=4K)
        llm_model: Gemini model for scenario/animation generation
        tts_model: Gemini TTS model for voice-over generation
        sections_limit: Maximum number of sections to process
        subject: Subject type (math, chemistry, physics, cs)
    """
    # Create unified output directory structure
    request_id = hashlib.md5((problem_text + str(os.urandom(8))).encode()).hexdigest()[:12]
    # Output goes to root/output folder
    base_output_dir = Path(__file__).parent.parent.parent / "output"
    request_output_dir = base_output_dir / request_id
    request_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🎬 VIDEO GENERATION STARTED")
    print("="*60)
    print(f"📁 Output folder: output/{request_id}/")
    print(f"📚 Subject: {subject.upper()}")
    print(f"🤖 LLM Model: {llm_model}")
    if generate_voiceover:
        print(f"🎙️ TTS Model: {tts_model}")
    print(f"📊 Quality: {quality}")
    print(f"📋 Sections Limit: {sections_limit}")
    print("="*60 + "\n")
    
    # First generate the scenario
    print("📝 Step 1/3: Generating video scenario...")
    scenario_text = generate_video_scenario(problem_text, llm_model=llm_model, subject=subject)
    
    if scenario_text.startswith("Error"):
        print("❌ Failed to generate scenario\n")
        return scenario_text, [], "Failed to generate scenario", None, [], None
    
    print("✅ Scenario generated successfully\n")
    
    # Parse scenario JSON and extract voice-over texts
    print("🔍 Step 2/3: Parsing scenario structure...")
    scenario_data, voice_texts, extraction_error = parse_scenario_json(scenario_text)
    
    if extraction_error:
        print(f"❌ Parsing failed: {extraction_error}\n")
        return scenario_text, [], extraction_error, None, [], None
    
    print(f"✅ Found {len(scenario_data.get('sections', []))} sections with {len(voice_texts)} frames\n")
    
    if not voice_texts:
        return scenario_text, [], "No voice-over texts found in scenario", None, [], None
    
    # Status messages initialization
    status_messages = []
    status_messages.append(f"📁 Output directory: output/{request_id}/\n")
    
    if not generate_voiceover:
        status_messages.append("🔇 Voice-over generation disabled\n")
    
    # Persist scenario JSON
    updated_scenario_json = scenario_text
    scenario_file_message = ""
    try:
        if scenario_data is not None:
            updated_scenario_json = json.dumps(scenario_data, ensure_ascii=False, indent=2)
            scenario_path = request_output_dir / "scenario.json"
            scenario_path.write_text(updated_scenario_json, encoding="utf-8")
            scenario_file_message = f"📝 Scenario saved: output/{request_id}/scenario.json"
    except Exception as scenario_error:
        scenario_file_message = f"⚠️ Failed to save scenario JSON: {scenario_error}"

    if scenario_file_message:
        status_messages.append(scenario_file_message)

    # Generate and render Manim scenes if requested (one at a time)
    # TTS generation moved here - after each scene renders successfully
    manim_files = []
    video_files = []
    audio_results = []
    final_video_path = None  # Initialize to prevent UnboundLocalError
    
    if generate_manim and scenario_data:
        status_messages.append(f"\n🤖 Using LLM model: {llm_model}")
        status_messages.append("\n🎬 Generating and rendering Manim animations (one at a time)...")
        
        print("\n" + "="*60)
        print("🎬 Step 3/3: Generating and Rendering Scenes")
        print("="*60)
        
        if generate_voiceover:
            status_messages.append(f"🎙️ Using TTS model: {tts_model}")
            print(f"🎙️ Voice-over enabled (TTS: {tts_model})")
        else:
            print("🔇 Voice-over disabled")
        print("")
        
        manim_dir = request_output_dir / "manim_scenes"
        manim_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS client if needed
        tts_client = None
        if generate_voiceover:
            tts_client = initialize_tts_client()
            if tts_client is None:
                status_messages.append("⚠️ Failed to initialize TTS client - voice-overs disabled")
                generate_voiceover = False
        
        # Track previous scene code for context continuity
        previous_scene_code = None
        scene_index = 0
        sections_processed = 0
        
        for section in scenario_data.get('sections', []):
            # Apply sections limit
            if sections_processed >= sections_limit:
                status_messages.append(f"\n⏹️ Reached section limit ({sections_limit}), stopping generation...")
                break
            
            section_name = section.get('section_name', 'Unknown Section')
            sections_processed += 1
            
            for frame in section.get('key_frames', []):
                frame_name = frame.get('name', 'Unknown Frame')
                animation_prompt = frame.get('animation_prompt', '')
                # Try both field names for compatibility
                voice_over_length = frame.get('estimate_length_timestamp') or frame.get('voice_over_length_timestamp', '00:05')
                
                if not animation_prompt:
                    continue
                
                # Parse timestamp to seconds and add 0.5s padding to prevent audio cutoff
                try:
                    parts = voice_over_length.split(':')
                    duration_seconds = int(parts[0]) * 60 + int(parts[1]) + 0.5  # Add 0.5s padding
                except (ValueError, IndexError):
                    duration_seconds = 5.5  # Default fallback with padding
                
                safe_filename = sanitize_class_name(f"{section_name}_{frame_name}")
                
                # ============================================================
                # STEP 1: Generate Manim code with previous scene context
                # ============================================================
                print(f"🔨 [{scene_index + 1}] Generating scene: {safe_filename}")
                status_messages.append(f"\n🔨 [{scene_index + 1}] Generating: {safe_filename}...")
                
                manim_code, error = generate_manim_scene(
                    section_name,
                    frame_name,
                    animation_prompt,
                    duration_seconds,
                    llm_model=llm_model,
                    previous_scene_code=previous_scene_code
                )
                
                if error:
                    print(f"   ❌ Generation failed: {error}")
                    status_messages.append(f"❌ Generation failed: {error}")
                    continue
                
                print("   ✅ Code generated")
                
                # Save to file
                manim_file_path = manim_dir / f"{safe_filename}.py"
                
                try:
                    manim_file_path.write_text(manim_code, encoding='utf-8')
                    status_messages.append(f"✅ Script saved: {safe_filename}.py")
                    print("   💾 Script saved")
                except Exception as write_error:
                    status_messages.append(f"❌ Failed to save script: {write_error}")
                    print(f"   ❌ Failed to save: {write_error}")
                    continue
                
                # ============================================================
                # STEP 2: Immediately render the scene
                # ============================================================
                print("   ⏳ Rendering...")
                status_messages.append(f"⏳ Rendering: {safe_filename}...")
                
                video_path, render_error = render_manim_scene(
                    str(manim_file_path),
                    safe_filename,
                    request_output_dir,
                    quality=quality
                )
                
                if render_error:
                    status_messages.append(f"❌ Render failed: {render_error}")
                    print(f"   ❌ Render failed: {render_error}")
                    # Don't use failed scene as context
                else:
                    status_messages.append(f"✅ Rendered successfully: {safe_filename}.mp4")
                    print("   ✅ Render complete")
                    
                    # ============================================================
                    # STEP 3: Generate audio AFTER successful render
                    # ============================================================
                    if generate_voiceover and tts_client:
                        # Get voice-over text for this frame
                        voice_text = frame.get('voice_over_text', '')
                        
                        if voice_text:
                            # Add rate limiting delay
                            if audio_results:  # Not first audio
                                time.sleep(2.5)  # 2.5s delay to avoid quota errors
                            
                            print("   🎙️ Generating voice-over...")
                            status_messages.append(f"🎙️ Generating audio for: {frame_name}...")
                            
                            audio_path, message, duration_secs, audio_bytes = generate_audio_from_text(
                                voice_text,
                                f"{section_name.lower().replace(' ', '_')}_{frame_name.lower().replace(' ', '_')}",
                                request_output_dir,
                                client=tts_client,
                                tts_model=tts_model
                            )
                            
                            if audio_path:
                                status_messages.append(f"✅ Audio: {message}")
                                print("   ✅ Audio generated")
                                
                                # Calculate timestamp
                                duration_timestamp = None
                                if duration_secs:
                                    minutes = int(duration_secs // 60)
                                    seconds = int(duration_secs % 60)
                                    duration_timestamp = f"{minutes:02d}:{seconds:02d}"
                                
                                audio_results.append({
                                    'section': section_name,
                                    'frame_name': frame_name,
                                    'text': voice_text,
                                    'audio_path': audio_path,
                                    'duration_seconds': duration_secs,
                                    'duration_timestamp': duration_timestamp,
                                    'audio_bytes': audio_bytes
                                })
                            else:
                                status_messages.append(f"❌ Audio failed: {message}")
                                print(f"   ❌ Audio failed: {message}")
                    
                    # ============================================================
                    # STEP 4: Store successful scene for context and tracking
                    # ============================================================
                    previous_scene_code = manim_code
                    
                    manim_files.append({
                        'section': section_name,
                        'frame': frame_name,
                        'path': str(manim_file_path),
                        'class_name': safe_filename,
                        'duration': duration_seconds
                    })
                    
                    video_files.append({
                        'section': section_name,
                        'frame': frame_name,
                        'video_path': video_path,
                        'class_name': safe_filename,
                        'index': scene_index
                    })
                
                scene_index += 1
        
        if video_files:
            status_messages.append(f"\n🎉 Successfully generated and rendered {len(video_files)} scenes in output/{request_id}/")
            
            if video_files:
                status_messages.append(f"\n🎬 Rendered {len(video_files)} videos in output/{request_id}/videos/")
                
                # Combine videos with audio if voiceover was generated
                final_video_path = None
                if generate_voiceover and audio_results:
                    print("\n" + "="*60)
                    print("🎙️ Step 3/3: Combining videos with audio...")
                    print("="*60)
                    status_messages.append("\n🎙️ Combining videos with voice-overs...")
                    
                    videos_with_audio_dir = request_output_dir / "videos_with_audio"
                    videos_with_audio_dir.mkdir(exist_ok=True)
                    
                    videos_with_audio = []
                    
                    for idx, video_info in enumerate(video_files, 1):
                        # Find matching audio
                        matching_audio = None
                        for audio_result in audio_results:
                            if (audio_result['section'] == video_info['section'] and 
                                audio_result['frame_name'] == video_info['frame']):
                                matching_audio = audio_result
                                break
                        
                        if matching_audio:
                            print(f"🔊 [{idx}/{len(video_files)}] Adding audio to {video_info['class_name']}...")
                            output_with_audio = videos_with_audio_dir / f"{video_info['class_name']}_with_audio.mp4"
                            success, error = combine_video_with_audio(
                                video_info['video_path'],
                                matching_audio['audio_path'],
                                output_with_audio,
                                audio_timing='auto'
                            )
                            
                            if success:
                                videos_with_audio.append(str(output_with_audio))
                                status_messages.append(f"✅ Combined audio: {video_info['class_name']}")
                                print("   ✅ Success")
                            else:
                                status_messages.append(f"⚠️ Audio combination failed for {video_info['class_name']}: {error}")
                                videos_with_audio.append(video_info['video_path'])
                                print("   ⚠️ Failed, using video without audio")
                        else:
                            # No audio for this video, use original
                            videos_with_audio.append(video_info['video_path'])
                    
                    # Concatenate all videos
                    if videos_with_audio:
                        print(f"\n🎞️ Concatenating {len(videos_with_audio)} videos into final output...")
                        print("   Videos to concatenate:")
                        for i, v in enumerate(videos_with_audio, 1):
                            print(f"   {i}. {Path(v).name}")
                        
                        status_messages.append("\n🎞️ Creating final combined video...")
                        final_video_path = request_output_dir / "final_video.mp4"
                        
                        success, error = concatenate_videos(videos_with_audio, final_video_path)
                        
                        if success:
                            status_messages.append(f"✅ Final video created: output/{request_id}/final_video.mp4")
                            print(f"✅ Final video ready: output/{request_id}/final_video.mp4")
                            
                            # Verify audio stream exists
                            try:
                                verify_cmd = [
                                    'ffprobe', '-v', 'error',
                                    '-select_streams', 'a',
                                    '-show_entries', 'stream=codec_name',
                                    '-of', 'default=noprint_wrappers=1:nokey=1',
                                    str(final_video_path)
                                ]
                                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, check=True)
                                if verify_result.stdout.strip():
                                    print(f"   ✅ Audio stream verified: {verify_result.stdout.strip()}")
                                else:
                                    print("   ⚠️ Warning: No audio stream detected in final video")
                            except Exception as verify_error:
                                print(f"   ⚠️ Could not verify audio stream: {verify_error}")
                        else:
                            status_messages.append(f"❌ Video concatenation failed: {error}")
                            print(f"❌ Concatenation failed: {error}")
                            final_video_path = None
                else:
                    # No voiceover, just concatenate videos
                    print("\n" + "="*60)
                    print("🎞️ Step 3/3: Creating final video (no audio)...")
                    print("="*60)
                    status_messages.append("\n🎞️ Creating final combined video...")
                    final_video_path = request_output_dir / "final_video.mp4"
                    
                    video_paths = [v['video_path'] for v in video_files]
                    success, error = concatenate_videos(video_paths, final_video_path)
                    
                    if success:
                        status_messages.append(f"✅ Final video created: output/{request_id}/final_video.mp4")
                        print(f"✅ Final video ready: output/{request_id}/final_video.mp4")
                    else:
                        status_messages.append(f"❌ Video concatenation failed: {error}")
                        final_video_path = None

    # Generate combined audio preview (optional - not displayed in UI anymore)
    combined_audio_value = None
    
    # Print final summary
    print("\n" + "="*60)
    print("✅ VIDEO GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 Output folder: output/{request_id}/")
    print(f"🎬 Scenes generated: {len(video_files)}")
    if generate_voiceover:
        print(f"🎙️ Audio tracks: {len(audio_results)}")
    if final_video_path:
        print(f"🎥 Final video: output/{request_id}/final_video.mp4")
    print("="*60 + "\n")
    
    final_status = f"Generated {len(audio_results)} audio tracks:\n" + "\n".join(status_messages)

    return updated_scenario_json, audio_results, final_status, combined_audio_value, manim_files, final_video_path

def generate_video_scenario(problem_text, llm_model="gemini-2.5-flash", subject="math"):
    """
    Uses Vertex AI or OpenRouter to generate a video scenario for an educational problem.
    
    Args:
        problem_text: The problem or concept to explain
        llm_model: Model to use (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro, or openrouter models)
        subject: Subject type (math, chemistry, physics, cs)
    """
    # Determine if using OpenRouter or Vertex AI
    use_openrouter = llm_model.startswith("openrouter/")
    
    if not use_openrouter:
        # Initialize Vertex AI
        if not initialize_vertex_ai():
            return "Error: Failed to initialize Vertex AI"
    
    # Map subject to template name
    subject_template_map = {
        "math": "video_scenario_math",
        "chemistry": "video_scenario_chemistry",
        "physics": "video_scenario_physics",
        "cs": "video_scenario_cs"
    }
    
    # Map subject to placeholder name in template
    subject_placeholder_map = {
        "math": "{math_problem}",
        "chemistry": "{chemistry_problem}",
        "physics": "{physics_problem}",
        "cs": "{cs_problem}"
    }
    
    template_name = subject_template_map.get(subject, "video_scenario_math")
    placeholder = subject_placeholder_map.get(subject, "{math_problem}")
    
    # Load the appropriate prompt template
    prompt_template = load_prompt_template(template_name)
    if prompt_template.startswith("Error:"):
        return prompt_template
    
    # Format the prompt with the problem text
    formatted_prompt = prompt_template.replace(placeholder, problem_text)
    
    # Add explicit JSON formatting instruction
    formatted_prompt += "\n\nIMPORTANT: Return ONLY valid JSON without any markdown code blocks. Ensure all JSON syntax is correct with proper commas and quotes."
    
    try:
        if use_openrouter:
            # Use OpenRouter API
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return "Error: OPENROUTER_API_KEY not found"
            
            # Extract model name (remove "openrouter/" prefix)
            model_name = llm_model.replace("openrouter/", "")
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost:7865",
                "X-Title": "Alfa Video Generator"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 8192
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON from markdown if present
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            return content
            
        else:
            # Use Vertex AI Gemini model with JSON mode
            model = GenerativeModel(llm_model)
            
            # Generate content with JSON output
            response = model.generate_content(
                formatted_prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                    response_mime_type="application/json"
                )
            )
            
            return response.text
        
    except Exception as e:
        return f"Error generating scenario: {str(e)}"

def create_gradio_app():
    """
    Creates and returns the Gradio interface for video scenario generation with audio.
    """
    with gr.Blocks(title="ALFA - Educational Video Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# � ALFA")
        gr.Markdown("**From problem to AI generated explainer animations** - Generate educational videos with voice-overs for Math, Chemistry, Physics, and Computer Science!")
        
        with gr.Row():
            with gr.Column(scale=2):
                subject_dropdown = gr.Dropdown(
                    label="📚 Subject",
                    choices=[
                        ("Mathematics", "math"),
                        ("Chemistry", "chemistry"),
                        ("Physics", "physics"),
                        ("Computer Science", "cs")
                    ],
                    value="math",
                    interactive=True,
                    info="Select the subject area for your problem"
                )
                
                problem_input = gr.Textbox(
                    label="Problem or Concept",
                    placeholder="Enter your problem or concept to explain (e.g., 'Explain binary search algorithm' or 'Balance the equation: H₂ + O₂ → H₂O')",
                    lines=3,
                    max_lines=5
                )
                
                quality_dropdown = gr.Dropdown(
                    label="Video Quality",
                    choices=[
                        ("Low (480p)", "l"),
                        ("Medium (720p)", "m"),
                        ("High (1080p)", "h"),
                        ("4K (2160p)", "k")
                    ],
                    value="m",
                    interactive=True
                )
                
                gr.Markdown("### Sections Limit")
                sections_limit_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=10,
                    step=1,
                    info="Limit generation to first N sections (useful for testing)"
                )
                
                generate_voiceover_checkbox = gr.Checkbox(
                    label="🎙️ Generate Voice-over",
                    value=True,
                    interactive=True,
                    info="Enable text-to-speech narration"
                )
                
                with gr.Row():
                    llm_model_dropdown = gr.Dropdown(
                        label="🤖 LLM Model",
                        choices=[
                            ("Gemini 2.5 Flash Lite (Vertex AI)", "gemini-2.5-flash-lite"),
                            ("Gemini 2.5 Flash (Vertex AI)", "gemini-2.5-flash"),
                            ("Gemini 2.5 Pro (Vertex AI)", "gemini-2.5-pro"),
                            ("──────────", None),
                            ("GPT-4o (OpenRouter)", "openrouter/openai/gpt-4o"),
                            ("GPT-4o Mini (OpenRouter)", "openrouter/openai/gpt-4o-mini"),
                            ("Claude 3.5 Sonnet (OpenRouter)", "openrouter/anthropic/claude-3.5-sonnet"),
                            ("Claude 3 Opus (OpenRouter)", "openrouter/anthropic/claude-3-opus"),
                            ("Gemini Pro 1.5 (OpenRouter)", "openrouter/google/gemini-pro-1.5"),
                            ("DeepSeek Chat (OpenRouter)", "openrouter/deepseek/deepseek-chat")
                        ],
                        value="gemini-2.5-flash",
                        interactive=True,
                        info="Model for generating scenarios and animation scripts"
                    )
                    
                    tts_model_dropdown = gr.Dropdown(
                        label="🎙️ Voice Model (Gemini TTS)",
                        choices=[
                            ("Gemini 2.5 Flash", "gemini-2.5-flash-tts"),
                            ("Gemini 2.5 Pro", "gemini-2.5-pro-tts")
                        ],
                        value="gemini-2.5-flash-tts",
                        interactive=True,
                        info="Model for voice-over generation"
                    )

                generate_audio_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=3):
                final_video_player = gr.Video(
                    label="Final Video",
                    interactive=False,
                    visible=False
                )
                
                scenario_output = gr.Textbox(
                    label="Video Scenario with Keyframes",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )

                audio_status = gr.Textbox(
                    label="Generation Status",
                    lines=5,
                    max_lines=15,
                    interactive=False,
                    visible=False
                )
        
        # Example problems
        gr.Markdown("### 📚 Try these example problems:")
        gr.Examples(
            examples=[
                ["Find the slope of the line passing through points (2, 4) and (6, 12)"],
                ["Solve for x: 3x - 7 = 14"],
                ["Find the area of a circle with radius 5 cm"],
                ["What is the derivative of f(x) = x² + 3x - 2?"],
                ["Simplify: (2x + 3)(x - 4)"],
                ["Solve the quadratic equation: x² - 5x + 6 = 0"]
            ],
            inputs=[problem_input],
            cache_examples=False
        )
        
        def handle_video_generation_ui(subject, problem_text, quality, generate_voiceover, llm_model, tts_model, sections_limit):
            """Handle scenario generation with audio and Manim scripts for UI."""
            scenario_text, audio_results, status, combined_audio_value, manim_files, final_video_path = generate_scenario_with_audio_and_manim(
                problem_text,
                gap_keyframe_seconds=0.3,  # Fixed: shorter gap to prevent audio delays
                gap_section_seconds=0.5,   # Fixed: shorter gap between sections
                generate_manim=True,
                generate_voiceover=generate_voiceover,
                quality=quality,
                llm_model=llm_model,
                tts_model=tts_model,
                sections_limit=sections_limit,
                subject=subject
            )

            return [
                gr.update(value=str(final_video_path) if final_video_path else None, visible=final_video_path is not None),
                scenario_text,
                gr.update(value=status, visible=bool(status))
            ]
        
        # Event handlers
        generate_audio_btn.click(
            fn=handle_video_generation_ui,
            inputs=[subject_dropdown, problem_input, quality_dropdown, generate_voiceover_checkbox, llm_model_dropdown, tts_model_dropdown, sections_limit_slider],
            outputs=[final_video_player, scenario_output, audio_status],
            show_progress=True
        )
    
    return app

def main():
    """
    Main function to launch the Video Scenario Generator with Audio app.
    """
    print("🚀 Starting ALFA - Educational Video Generator...")
    
    app = create_gradio_app()
    
    # Launch the app
    app.launch(
        server_name="127.0.0.1",
        server_port=7865,
        show_api=True,
        share=False
    )

if __name__ == "__main__":
    main()
