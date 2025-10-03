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

PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)
PCM_CHANNELS = 1
PCM_SAMPLE_RATE = 24000

# Load environment variables
load_dotenv()

def initialize_vertex_ai():
    """
    Initialize Vertex AI with credentials from service account.
    """
    credentials_path = Path(__file__).parent / ".credentials" / "alfa_gcp_sa.json"
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


def generate_manim_scene(section_name, keyframe_name, animation_prompt, voice_over_duration, llm_model="gemini-2.5-flash"):
    """
    Use Vertex AI Gemini API to generate a Manim scene Python script for a keyframe.
    
    Args:
        section_name: Name of the section
        keyframe_name: Name of the keyframe
        animation_prompt: Prompt for the animation
        voice_over_duration: Duration of voice-over in seconds
        llm_model: Gemini model to use (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro)
    """
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
    
    # Add instruction for structured output
    formatted_prompt += "\n\nIMPORTANT: Return ONLY the Python code, without markdown code blocks or any additional text."
    
    try:
        # Initialize Vertex AI Gemini model with text output (not JSON for code)
        model = GenerativeModel(llm_model)
        
        # Generate content
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
        return None, f"Error generating Manim code with Vertex AI: {str(e)}"


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


def apply_fade(samples, sample_rate=PCM_SAMPLE_RATE, fade_duration=0.01):
    """Apply a short fade-in and fade-out to reduce clicks at boundaries."""
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
        if delay > 0:
            # If there's a delay, use adelay filter
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(video_path),
                '-i', str(audio_path),
                '-filter_complex',
                f'[1:a]adelay={int(delay * 1000)}|{int(delay * 1000)}[aout]',
                '-map', '0:v',
                '-map', '[aout]',
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                str(output_path)
            ]
        else:
            # No delay, just add audio directly
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(video_path),
                '-i', str(audio_path),
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',  # Copy video without re-encoding
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
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
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list,
            '-c', 'copy',  # Copy without re-encoding for speed
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
    credentials_path = Path(__file__).parent / ".credentials" / "alfa_gcp_sa.json"
    
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
    tts_model="gemini-2.5-flash-tts"
):
    """
    Generate a video scenario, create audio tracks, and generate Manim animation scripts.
    
    Args:
        problem_text: The math problem to solve
        gap_keyframe_seconds: Gap between keyframes in audio
        gap_section_seconds: Gap between sections in audio
        generate_manim: Whether to generate Manim animations
        generate_voiceover: Whether to generate voice-over audio
        quality: Video quality ('l'=480p, 'm'=720p, 'h'=1080p, 'k'=4K)
        llm_model: Gemini model for scenario/animation generation
        tts_model: Gemini TTS model for voice-over generation
    """
    # Create unified output directory structure
    request_id = hashlib.md5((problem_text + str(os.urandom(8))).encode()).hexdigest()[:12]
    base_output_dir = Path(__file__).parent / "output"
    request_output_dir = base_output_dir / request_id
    request_output_dir.mkdir(parents=True, exist_ok=True)
    
    # First generate the scenario
    scenario_text = generate_video_scenario(problem_text, llm_model=llm_model)
    
    if scenario_text.startswith("Error"):
        return scenario_text, [], "Failed to generate scenario", None, [], None
    
    # Parse scenario JSON and extract voice-over texts
    scenario_data, voice_texts, extraction_error = parse_scenario_json(scenario_text)
    
    if extraction_error:
        return scenario_text, [], extraction_error, None, [], None
    
    if not voice_texts:
        return scenario_text, [], "No voice-over texts found in scenario", None, [], None
    
    # Initialize TTS client only if generating voiceovers
    tts_client = None
    if generate_voiceover:
        tts_client = initialize_tts_client()
        if tts_client is None:
            return scenario_text, [], "Failed to initialize Text-to-Speech client", None, [], None
    
    # Generate audio for each voice-over
    audio_results = []
    status_messages = []
    
    status_messages.append(f"📁 Output directory: output/{request_id}/\n")
    
    if not generate_voiceover:
        status_messages.append("🔇 Voice-over generation disabled\n")
    else:
        status_messages.append(f"🎙️ Using TTS model: {tts_model}\n")
        
        for idx, voice_data in enumerate(voice_texts):
            # Add rate limiting: wait between requests to avoid quota errors
            if idx > 0:
                time.sleep(1.5)  # Wait 1.5 seconds between TTS requests
            
            audio_path, message, duration_seconds, audio_bytes = generate_audio_from_text(
                voice_data['text'], 
                voice_data['filename_prefix'],
                request_output_dir,
                tts_client,
                tts_model=tts_model
            )
            
            if audio_path:
                duration_timestamp = format_seconds_to_timestamp(duration_seconds)

                # Update scenario JSON with the measured duration
                if duration_timestamp:
                    try:
                        scenario_data['sections'][voice_data['section_index']]['key_frames'][voice_data['frame_index']]['voice_over_length_timestamp'] = duration_timestamp
                    except (IndexError, KeyError, TypeError):
                        print("Warning: Unable to write voice_over_length_timestamp for one of the keyframes.")

                audio_results.append({
                    'section': voice_data['section'],
                    'frame_name': voice_data['frame_name'],
                    'timestamp': voice_data['timestamp'],
                    'text': voice_data['text'],
                    'audio_path': audio_path,
                    'duration_seconds': duration_seconds,
                    'duration_timestamp': duration_timestamp,
                    'audio_bytes': audio_bytes
                })
                status_appendix = f" (duration {duration_timestamp})" if duration_timestamp else ""
                status_messages.append(f"✅ {voice_data['frame_name']}: {message}{status_appendix}")
            else:
                status_messages.append(f"❌ {voice_data['frame_name']}: {message}")
    
    combined_audio_value = None
    if audio_results:
        combined_array, combine_error = combine_audio_arrays(
            audio_results,
            gap_keyframe_seconds=gap_keyframe_seconds,
            gap_section_seconds=gap_section_seconds,
            sample_rate=PCM_SAMPLE_RATE
        )

        if combine_error:
            status_messages.append(f"⚠️ Unable to combine audio tracks: {combine_error}")
        else:
            combined_audio_value = numpy_audio_to_gradio_value(combined_array, sample_rate=PCM_SAMPLE_RATE)
            if combined_audio_value is not None:
                status_messages.append("🎧 Combined audio track ready for playback.")

    # Persist updated scenario JSON with voice-over durations
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

    # Generate Manim scene files if requested
    manim_files = []
    if generate_manim and scenario_data:
        status_messages.append(f"\n🤖 Using LLM model: {llm_model}")
        status_messages.append("\n🎬 Generating Manim animation scripts...")
        
        manim_dir = request_output_dir / "manim_scenes"
        manim_dir.mkdir(parents=True, exist_ok=True)
        
        for section in scenario_data.get('sections', []):
            section_name = section.get('section_name', 'Unknown Section')
            
            for frame in section.get('key_frames', []):
                frame_name = frame.get('name', 'Unknown Frame')
                animation_prompt = frame.get('animation_prompt', '')
                voice_over_length = frame.get('voice_over_length_timestamp', '00:05')
                
                if not animation_prompt:
                    continue
                
                # Parse timestamp to seconds
                try:
                    parts = voice_over_length.split(':')
                    duration_seconds = int(parts[0]) * 60 + int(parts[1])
                except (ValueError, IndexError):
                    duration_seconds = 5  # Default fallback
                
                # Generate Manim code
                manim_code, error = generate_manim_scene(
                    section_name,
                    frame_name,
                    animation_prompt,
                    duration_seconds,
                    llm_model=llm_model
                )
                
                if error:
                    status_messages.append(f"⚠️ Manim generation failed for {frame_name}: {error}")
                    continue
                
                # Save to file
                safe_filename = sanitize_class_name(f"{section_name}_{frame_name}")
                manim_file_path = manim_dir / f"{safe_filename}.py"
                
                try:
                    manim_file_path.write_text(manim_code, encoding='utf-8')
                    manim_files.append({
                        'section': section_name,
                        'frame': frame_name,
                        'path': str(manim_file_path),
                        'class_name': safe_filename,
                        'duration': duration_seconds
                    })
                    status_messages.append(f"✅ Manim script: {safe_filename}.py")
                except Exception as write_error:
                    status_messages.append(f"⚠️ Failed to save {safe_filename}.py: {write_error}")
        
        if manim_files:
            status_messages.append(f"\n🎉 Generated {len(manim_files)} Manim scene files in output/{request_id}/manim_scenes/")
            
            # Render Manim scenes to video
            status_messages.append("\n🎥 Rendering animations to video...")
            video_files = []
            
            for idx, manim_info in enumerate(manim_files):
                script_path = manim_info['path']
                class_name = manim_info['class_name']
                
                status_messages.append(f"⏳ Rendering {class_name}...")
                
                video_path, render_error = render_manim_scene(
                    script_path,
                    class_name,
                    request_output_dir,
                    quality=quality
                )
                
                if render_error:
                    status_messages.append(f"❌ Render failed for {class_name}: {render_error}")
                else:
                    video_info = {
                        'section': manim_info['section'],
                        'frame': manim_info['frame'],
                        'video_path': video_path,
                        'class_name': class_name,
                        'index': idx
                    }
                    video_files.append(video_info)
                    status_messages.append(f"✅ Video rendered: {class_name}.mp4")
            
            if video_files:
                status_messages.append(f"\n🎬 Rendered {len(video_files)} videos in output/{request_id}/videos/")
                
                # Combine videos with audio if voiceover was generated
                final_video_path = None
                if generate_voiceover and audio_results:
                    status_messages.append("\n🎙️ Combining videos with voice-overs...")
                    
                    videos_with_audio_dir = request_output_dir / "videos_with_audio"
                    videos_with_audio_dir.mkdir(exist_ok=True)
                    
                    videos_with_audio = []
                    
                    for video_info in video_files:
                        # Find matching audio
                        matching_audio = None
                        for audio_result in audio_results:
                            if (audio_result['section'] == video_info['section'] and 
                                audio_result['frame_name'] == video_info['frame']):
                                matching_audio = audio_result
                                break
                        
                        if matching_audio:
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
                            else:
                                status_messages.append(f"⚠️ Audio combination failed for {video_info['class_name']}: {error}")
                                videos_with_audio.append(video_info['video_path'])
                        else:
                            # No audio for this video, use original
                            videos_with_audio.append(video_info['video_path'])
                    
                    # Concatenate all videos
                    if videos_with_audio:
                        status_messages.append("\n🎞️ Creating final combined video...")
                        final_video_path = request_output_dir / "final_video.mp4"
                        
                        success, error = concatenate_videos(videos_with_audio, final_video_path)
                        
                        if success:
                            status_messages.append(f"✅ Final video created: output/{request_id}/final_video.mp4")
                        else:
                            status_messages.append(f"❌ Video concatenation failed: {error}")
                            final_video_path = None
                else:
                    # No voiceover, just concatenate videos
                    status_messages.append("\n🎞️ Creating final combined video...")
                    final_video_path = request_output_dir / "final_video.mp4"
                    
                    video_paths = [v['video_path'] for v in video_files]
                    success, error = concatenate_videos(video_paths, final_video_path)
                    
                    if success:
                        status_messages.append(f"✅ Final video created: output/{request_id}/final_video.mp4")
                    else:
                        status_messages.append(f"❌ Video concatenation failed: {error}")
                        final_video_path = None

    final_status = f"Generated {len(audio_results)} audio tracks out of {len(voice_texts)} voice-overs:\n" + "\n".join(status_messages)

    return updated_scenario_json, audio_results, final_status, combined_audio_value, manim_files, final_video_path

def generate_video_scenario(problem_text, llm_model="gemini-2.5-flash"):
    """
    Uses Vertex AI Gemini API to generate a video scenario for a math problem.
    
    Args:
        problem_text: The math problem to solve
        llm_model: Gemini model to use (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro)
    """
    # Initialize Vertex AI
    if not initialize_vertex_ai():
        return "Error: Failed to initialize Vertex AI"
    
    # Load the appropriate prompt template
    prompt_template = load_prompt_template("video_scenario")
    if prompt_template.startswith("Error:"):
        return prompt_template
    
    # Format the prompt with the math problem using replace instead of format
    formatted_prompt = prompt_template.replace("{math_problem}", problem_text)
    
    # Add explicit JSON formatting instruction
    formatted_prompt += "\n\nIMPORTANT: Return ONLY valid JSON without any markdown code blocks. Ensure all JSON syntax is correct with proper commas and quotes."
    
    try:
        # Initialize Vertex AI Gemini model with JSON mode
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
        return f"Error generating scenario with Vertex AI: {str(e)}"

def create_gradio_app():
    """
    Creates and returns the Gradio interface for video scenario generation with audio.
    """
    with gr.Blocks(title="Math Video Animation Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬🎵 Math Video Animation Generator")
        gr.Markdown("Enter a math problem and get a complete animated video with voice-overs!")
        
        with gr.Row():
            with gr.Column(scale=2):
                problem_input = gr.Textbox(
                    label="Math Problem",
                    placeholder="Enter your math problem here (e.g., 'Solve for x: 2x + 5 = 15')",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Row():
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
                    
                    generate_voiceover_checkbox = gr.Checkbox(
                        label="Generate Voice-over",
                        value=True,
                        interactive=True
                    )
                
                with gr.Row():
                    llm_model_dropdown = gr.Dropdown(
                        label="🤖 LLM Model (Vertex AI)",
                        choices=[
                            ("Gemini 2.5 Flash Lite", "gemini-2.5-flash-lite"),
                            ("Gemini 2.5 Flash", "gemini-2.5-flash"),
                            ("Gemini 2.5 Pro", "gemini-2.5-pro")
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

                gap_keyframe_slider = gr.Slider(
                    label="Gap between keyframes (seconds)",
                    minimum=0.0,
                    maximum=3.0,
                    value=0.5,
                    step=0.1
                )

                gap_section_slider = gr.Slider(
                    label="Gap between sections (seconds)",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.1
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

                combined_audio_player = gr.Audio(
                    label="Combined Audio Preview",
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
        
        def handle_video_generation_ui(problem_text, quality, generate_voiceover, llm_model, tts_model, gap_keyframe, gap_section):
            """Handle scenario generation with audio and Manim scripts for UI."""
            scenario_text, audio_results, status, combined_audio_value, manim_files, final_video_path = generate_scenario_with_audio_and_manim(
                problem_text,
                gap_keyframe_seconds=gap_keyframe,
                gap_section_seconds=gap_section,
                generate_manim=True,
                generate_voiceover=generate_voiceover,
                quality=quality,
                llm_model=llm_model,
                tts_model=tts_model
            )

            return [
                gr.update(value=str(final_video_path) if final_video_path else None, visible=final_video_path is not None),
                scenario_text,
                gr.update(value=status, visible=bool(status)),
                gr.update(value=combined_audio_value, visible=combined_audio_value is not None and generate_voiceover)
            ]
        
        # Event handlers
        generate_audio_btn.click(
            fn=handle_video_generation_ui,
            inputs=[problem_input, quality_dropdown, generate_voiceover_checkbox, llm_model_dropdown, tts_model_dropdown, gap_keyframe_slider, gap_section_slider],
            outputs=[final_video_player, scenario_output, audio_status, combined_audio_player],
            show_progress=True
        )
    
    return app

def main():
    """
    Main function to launch the Video Scenario Generator with Audio app.
    """
    print("🚀 Starting Math Video Scenario Generator with Audio...")
    
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
