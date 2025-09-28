import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path

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
    Creates and returns the Gradio interface for video scenario generation.
    """
    with gr.Blocks(title="Math Video Scenario Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Math Video Scenario Generator")
        gr.Markdown("Enter a math problem and get a detailed video scenario with keyframes and animations!")
        
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
                
                generate_btn = gr.Button("Generate Video Scenario", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                scenario_output = gr.Textbox(
                    label="Video Scenario with Keyframes",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Example problems
        gr.Markdown("### 📚 Try these example problems:")
        gr.Examples(
            examples=[
                ["Solve for x: 3x - 7 = 14"],
                ["Find the area of a circle with radius 5 cm"],
                ["What is the derivative of f(x) = x² + 3x - 2?"],
                ["Simplify: (2x + 3)(x - 4)"],
                ["Solve the quadratic equation: x² - 5x + 6 = 0"],
                ["Find the slope of the line passing through points (2, 4) and (6, 12)"]
            ],
            inputs=[problem_input],
            cache_examples=False
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_video_scenario,
            inputs=[problem_input, scenario_type],
            outputs=[scenario_output],
            show_progress=True
        )
        
        # Allow Enter key to submit
        problem_input.submit(
            fn=generate_video_scenario,
            inputs=[problem_input, scenario_type],
            outputs=[scenario_output],
            show_progress=True
        )
    
    return app

def main():
    """
    Main function to launch the Video Scenario Generator app.
    """
    print("🚀 Starting Math Video Scenario Generator...")
    
    app = create_gradio_app()
    
    # Launch the app
    app.launch(
        server_name="127.0.0.1",
        server_port=7863,
        show_api=True,
        share=False
    )

if __name__ == "__main__":
    main()
