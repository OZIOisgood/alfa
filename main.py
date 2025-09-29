import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from google.cloud import texttospeech
from google.oauth2 import service_account
from io import BytesIO
from mutagen.mp3 import MP3
import hashlib
import re

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

def generate_audio_from_text(text, filename_prefix, client=None):
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
        
        # Configure audio format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
            volume_gain_db=0.0
        )
        
        # Generate speech
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_bytes = response.audio_content

        # Determine duration directly from in-memory bytes before writing to disk
        duration_seconds = None
        try:
            duration_seconds = MP3(BytesIO(audio_bytes)).info.length
        except Exception as duration_error:
            print(f"Warning: Unable to determine audio duration: {duration_error}")

        # Save audio file
        audio_dir = Path(__file__).parent / "audio_output"
        audio_dir.mkdir(exist_ok=True)
        
        # Create filename with hash to avoid conflicts
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        filename = f"{filename_prefix}_{text_hash}.mp3"
        audio_path = audio_dir / filename
        
        with open(audio_path, "wb") as out:
            out.write(audio_bytes)
        
        return str(audio_path), f"Audio generated successfully: {filename}", duration_seconds
        
    except Exception as e:
        return None, f"Error generating audio: {str(e)}", None

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

def generate_scenario_with_audio(problem_text, scenario_type="video_scenario"):
    """
    Generate a video scenario and create audio tracks for all voice-overs.
    """
    # First generate the scenario
    scenario_text = generate_video_scenario(problem_text, scenario_type)
    
    if scenario_text.startswith("Error"):
        return scenario_text, [], "Failed to generate scenario"
    
    # Parse scenario JSON and extract voice-over texts
    scenario_data, voice_texts, extraction_error = parse_scenario_json(scenario_text)
    
    if extraction_error:
        return scenario_text, [], extraction_error
    
    if not voice_texts:
        return scenario_text, [], "No voice-over texts found in scenario"
    
    # Initialize TTS client
    tts_client = initialize_tts_client()
    if tts_client is None:
        return scenario_text, [], "Failed to initialize Text-to-Speech client"
    
    # Generate audio for each voice-over
    audio_results = []
    status_messages = []
    
    for voice_data in voice_texts:
        audio_path, message, duration_seconds = generate_audio_from_text(
            voice_data['text'], 
            voice_data['filename_prefix'], 
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
                'duration_timestamp': duration_timestamp
            })
            status_appendix = f" (duration {duration_timestamp})" if duration_timestamp else ""
            status_messages.append(f"✅ {voice_data['frame_name']}: {message}{status_appendix}")
        else:
            status_messages.append(f"❌ {voice_data['frame_name']}: {message}")
    
    # Persist updated scenario JSON with voice-over durations
    updated_scenario_json = scenario_text
    scenario_file_message = ""
    try:
        if scenario_data is not None:
            updated_scenario_json = json.dumps(scenario_data, ensure_ascii=False, indent=2)
            scenario_dir = Path(__file__).parent / "scenario_output"
            scenario_dir.mkdir(exist_ok=True)
            scenario_hash = hashlib.md5((problem_text + updated_scenario_json).encode()).hexdigest()[:8]
            scenario_path = scenario_dir / f"scenario_{scenario_hash}.json"
            scenario_path.write_text(updated_scenario_json, encoding="utf-8")
            scenario_file_message = f"📝 Scenario saved: {scenario_path.name}"
    except Exception as scenario_error:
        scenario_file_message = f"⚠️ Failed to save scenario JSON: {scenario_error}"

    if scenario_file_message:
        status_messages.append(scenario_file_message)

    final_status = f"Generated {len(audio_results)} audio tracks out of {len(voice_texts)} voice-overs:\n" + "\n".join(status_messages)

    return updated_scenario_json, audio_results, final_status

def generate_video_scenario(problem_text, scenario_type="video_scenario"):
    """
    Uses OpenRouter API to generate a video scenario for a math problem.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment variables."
    
    # Load the appropriate prompt template
    prompt_template = load_prompt_template(scenario_type)
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
                
                scenario_type = gr.Dropdown(
                    choices=["video_scenario", "simple_explanation"],
                    value="video_scenario",
                    label="Scenario Type",
                    info="Choose the type of output you want"
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate Scenario Only", variant="secondary", size="lg")
                    generate_audio_btn = gr.Button("Generate Scenario + Audio", variant="primary", size="lg")
                
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
        
        # Audio section that will be populated dynamically
        audio_section = gr.Column(visible=False)
        with audio_section:
            gr.Markdown("### 🎵 Generated Audio Tracks")
            audio_description = gr.HTML(value="", label="Audio Track Information")
            
            # Pre-create multiple audio components (we'll show/hide as needed)
            audio_players = []
            for i in range(10):  # Support up to 10 audio tracks
                audio_players.append(
                    gr.Audio(
                        label=f"Audio Track {i+1}",
                        interactive=False,
                        visible=False
                    )
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
        
        def handle_audio_generation_ui(problem_text, scenario_type):
            """Handle scenario generation with audio for UI."""
            scenario_text, audio_results, status = generate_scenario_with_audio(problem_text, scenario_type)
            
            # Create description HTML
            audio_html = ""
            if audio_results:
                audio_html = "<div style='margin-bottom: 20px;'>"
                for i, audio_data in enumerate(audio_results):
                    audio_html += f"""
                    <div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 8px; background: #f9f9f9;'>
                        <h4 style='margin-top: 0; color: #333;'>{audio_data['section']} - {audio_data['frame_name']}</h4>
                        <p style='margin: 5px 0; color: #666;'><strong>Timestamp:</strong> {audio_data['timestamp']}</p>
                        <p style='margin: 5px 0; color: #666;'><strong>Duration:</strong> {audio_data.get('duration_timestamp', 'Unknown')}</p>
                        <p style='margin: 10px 0; font-style: italic; color: #444;'>"{audio_data['text']}"</p>
                        <p style='margin: 5px 0; font-size: 12px; color: #888;'>🎵 Audio Track {i+1}</p>
                    </div>
                    """
                audio_html += "</div>"
            
            # Prepare outputs: scenario, status, description, section visibility, then audio players
            outputs = [
                scenario_text,
                gr.update(value=status, visible=bool(status)),  # audio_status
                audio_html,
                gr.update(visible=bool(audio_results))  # audio_section
            ]
            
            # Add audio player updates
            for i in range(10):
                if i < len(audio_results):
                    audio_data = audio_results[i]
                    label = f"{audio_data['section']} - {audio_data['frame_name']} ({audio_data['timestamp']})"
                    if audio_data.get('duration_timestamp'):
                        label += f" [{audio_data['duration_timestamp']}]"

                    outputs.append(gr.update(
                        value=audio_data['audio_path'],
                        label=label,
                        visible=True
                    ))
                else:
                    outputs.append(gr.update(value=None, visible=False))
            
            return outputs
        
        # Event handlers
        generate_btn.click(
            fn=generate_video_scenario,
            inputs=[problem_input, scenario_type],
            outputs=[scenario_output],
            show_progress=True
        )
        
        generate_audio_btn.click(
            fn=handle_audio_generation_ui,
            inputs=[problem_input, scenario_type],
            outputs=[scenario_output, audio_status, audio_description, audio_section] + audio_players,
            show_progress=True
        )
        
        # Allow Enter key to submit (scenario only)
        problem_input.submit(
            fn=generate_video_scenario,
            inputs=[problem_input, scenario_type],
            outputs=[scenario_output],
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
