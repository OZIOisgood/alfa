import os
import json
import wave
import subprocess
import requests
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from google.cloud import texttospeech
from google.oauth2 import service_account
import numpy as np
import hashlib
import re

PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)
PCM_CHANNELS = 1
PCM_SAMPLE_RATE = 24000

# Load environment variables
load_dotenv()

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


def generate_manim_scene(section_name, keyframe_name, animation_prompt, voice_over_duration):
    """
    Use OpenRouter API to generate a Manim scene Python script for a keyframe.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        return None, "Error: OPENROUTER_API_KEY not found"
    
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
    
    # OpenRouter API call
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:7865",
        "X-Title": "Manim Scene Generator"
    }
    
    data = {
        "model": "openai/gpt-4o",  # Use GPT-4 for better code generation
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 3000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        code = result['choices'][0]['message']['content']
        
        # Extract Python code from markdown if present
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


def render_manim_scene(script_path, class_name, output_dir, quality='m'):
    """
    Render a Manim scene to video using local manim command.
    
    Args:
        script_path: Path to the Python file containing the scene
        class_name: Name of the Scene class to render
        output_dir: Directory where output video should be saved
        quality: Quality setting (l=480p, m=720p, h=1080p, k=4K)
    
    Returns:
        (video_path, error_message) tuple
    """
    try:
        # Prepare output directory
        video_output_dir = Path(output_dir) / "videos"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare manim command with custom media directory
        quality_flag = f"-q{quality}"
        script_parent = Path(script_path).parent
        
        # Get current environment and ensure PATH is inherited
        import os
        import shutil as sh
        env = os.environ.copy()
        
        # Try to locate ffmpeg explicitly and add to PATH if found
        ffmpeg_path = sh.which("ffmpeg")
        if ffmpeg_path:
            ffmpeg_dir = str(Path(ffmpeg_path).parent)
            # Ensure ffmpeg directory is in PATH
            if ffmpeg_dir not in env.get("PATH", ""):
                env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
        
        # Run manim render command with custom media directory
        result = subprocess.run(
            ["manim", quality_flag, "--media_dir", str(script_parent / "media"), str(script_path), class_name],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per scene
            cwd=str(script_parent),  # Run from script directory
            env=env  # Inherit full environment including PATH with ffmpeg
        )
        
        if result.returncode != 0:
            # Check for common errors
            error_msg = result.stderr + result.stdout  # Combine both streams
            
            # Check for syntax warnings/errors in the generated code
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
        
        # Manim saves to {media_dir}/videos/{ScriptName}/{quality}/
        script_name = Path(script_path).stem
        quality_map = {'l': '480p15', 'm': '720p30', 'h': '1080p60', 'k': '2160p60'}
        quality_folder = quality_map.get(quality, '720p30')
        
        # Look for the rendered video
        media_dir = script_parent / "media" / "videos" / script_name / quality_folder
        
        if not media_dir.exists():
            return None, f"Output directory not found: {media_dir}"
        
        # Find the MP4 file
        video_files = list(media_dir.glob("*.mp4"))
        if not video_files:
            return None, f"No video file found in {media_dir}"
        
        # Copy to output directory
        source_video = video_files[0]
        dest_video = video_output_dir / f"{class_name}.mp4"
        
        # Copy file to output location
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

def generate_audio_from_text(text, filename_prefix, output_dir, client=None):
    """
    Generate audio from text using Google Cloud Text-to-Speech API.
    Uses Studio Q voice as primary, with fallback to standard voices.
    """
    if client is None:
        client = initialize_tts_client()
        if client is None:
            return None, "Failed to initialize TTS client"
    
    try:
        # Create synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use reliable standard voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-J",  # Standard male voice - reliable and clear
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
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
        return None, f"Error generating audio: {str(e)}", None, None

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

def generate_scenario_with_audio_and_manim(problem_text, gap_keyframe_seconds=0.5, gap_section_seconds=1.0, generate_manim=True):
    """
    Generate a video scenario, create audio tracks, and generate Manim animation scripts.
    """
    # Create unified output directory structure
    request_id = hashlib.md5((problem_text + str(os.urandom(8))).encode()).hexdigest()[:12]
    base_output_dir = Path(__file__).parent / "output"
    request_output_dir = base_output_dir / request_id
    request_output_dir.mkdir(parents=True, exist_ok=True)
    
    # First generate the scenario
    scenario_text = generate_video_scenario(problem_text)
    
    if scenario_text.startswith("Error"):
        return scenario_text, [], "Failed to generate scenario", None, []
    
    # Parse scenario JSON and extract voice-over texts
    scenario_data, voice_texts, extraction_error = parse_scenario_json(scenario_text)
    
    if extraction_error:
        return scenario_text, [], extraction_error, None, []
    
    if not voice_texts:
        return scenario_text, [], "No voice-over texts found in scenario", None, []
    
    # Initialize TTS client
    tts_client = initialize_tts_client()
    if tts_client is None:
        return scenario_text, [], "Failed to initialize Text-to-Speech client", None, []
    
    # Generate audio for each voice-over
    audio_results = []
    status_messages = []
    
    status_messages.append(f"📁 Output directory: output/{request_id}/\n")
    
    for voice_data in voice_texts:
        audio_path, message, duration_seconds, audio_bytes = generate_audio_from_text(
            voice_data['text'], 
            voice_data['filename_prefix'],
            request_output_dir,
            tts_client
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
                    duration_seconds
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
            
            for manim_info in manim_files:
                script_path = manim_info['path']
                class_name = manim_info['class_name']
                
                status_messages.append(f"⏳ Rendering {class_name}...")
                
                video_path, render_error = render_manim_scene(
                    script_path,
                    class_name,
                    request_output_dir,
                    quality='m'  # 720p
                )
                
                if render_error:
                    status_messages.append(f"❌ Render failed for {class_name}: {render_error}")
                else:
                    video_files.append({
                        'section': manim_info['section'],
                        'frame': manim_info['frame'],
                        'video_path': video_path,
                        'class_name': class_name
                    })
                    status_messages.append(f"✅ Video rendered: {class_name}.mp4")
            
            if video_files:
                status_messages.append(f"\n🎬 Rendered {len(video_files)} videos in output/{request_id}/videos/")

    final_status = f"Generated {len(audio_results)} audio tracks out of {len(voice_texts)} voice-overs:\n" + "\n".join(status_messages)

    return updated_scenario_json, audio_results, final_status, combined_audio_value, manim_files

def generate_video_scenario(problem_text):
    """
    Uses OpenRouter API to generate a video scenario for a math problem.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment variables."
    
    # Load the appropriate prompt template
    prompt_template = load_prompt_template("video_scenario")
    if prompt_template.startswith("Error:"):
        return prompt_template
    
    # Format the prompt with the math problem using replace instead of format
    formatted_prompt = prompt_template.replace("{math_problem}", problem_text)
    
    # OpenRouter API endpoint  
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:7863",  # For gradio
        "X-Title": "Math Video Scenario Generator"
    }
    
    # Using GPT-3.5-turbo for better structured output
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {
                "role": "user", 
                "content": formatted_prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Debug information
        print(f"Request URL: {url}")
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        response.raise_for_status()
        
        result = response.json()
        scenario = result['choices'][0]['message']['content']
        return scenario
        
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenRouter API: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"
    except KeyError as e:
        return f"Error parsing API response: {str(e)}\nFull response: {result if 'result' in locals() else 'No result'}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def create_gradio_app():
    """
    Creates and returns the Gradio interface for video scenario generation with audio.
    """
    with gr.Blocks(title="Math Video Scenario Generator with Audio", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬🎵 Math Video Scenario Generator with Audio")
        gr.Markdown("Enter a math problem and get a detailed video scenario with keyframes, animations, and generated audio tracks!")
        
        with gr.Row():
            with gr.Column(scale=2):
                problem_input = gr.Textbox(
                    label="Math Problem",
                    placeholder="Enter your math problem here (e.g., 'Solve for x: 2x + 5 = 15')",
                    lines=3,
                    max_lines=5
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

                generate_audio_btn = gr.Button("generate", variant="primary", size="lg")

            with gr.Column(scale=3):
                scenario_output = gr.Textbox(
                    label="Video Scenario with Keyframes",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    show_copy_button=True
                )

                audio_status = gr.Textbox(
                    label="Audio Generation Status",
                    lines=3,
                    max_lines=10,
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
        
        def handle_audio_generation_ui(problem_text, gap_keyframe, gap_section):
            """Handle scenario generation with audio and Manim scripts for UI."""
            scenario_text, audio_results, status, combined_audio_value, manim_files = generate_scenario_with_audio_and_manim(
                problem_text,
                gap_keyframe_seconds=gap_keyframe,
                gap_section_seconds=gap_section,
                generate_manim=True
            )

            return [
                scenario_text,
                gr.update(value=status, visible=bool(status)),
                gr.update(value=combined_audio_value, visible=combined_audio_value is not None)
            ]
        
        # Event handlers
        generate_audio_btn.click(
            fn=handle_audio_generation_ui,
            inputs=[problem_input, gap_keyframe_slider, gap_section_slider],
            outputs=[scenario_output, audio_status, combined_audio_player],
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
