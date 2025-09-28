import os
import requests
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def explain_math_problem(problem_text):
    """
    Uses OpenRouter API to explain a math problem step by step.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment variables."
    
    # OpenRouter API endpoint  
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:7862",  # For gradio
        "X-Title": "Math Problem Explainer"
    }
    
    # Using a cheap model like Llama 3.1 8B
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Explain math problems step by step in a clear and easy-to-understand way. Break down complex problems into smaller steps and provide reasoning for each step."
            },
            {
                "role": "user", 
                "content": f"Please explain this math problem step by step: {problem_text}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Debug information
        print(f"Request URL: {url}")
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        response.raise_for_status()
        
        result = response.json()
        explanation = result['choices'][0]['message']['content']
        return explanation
        
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenRouter API: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"
    except KeyError as e:
        return f"Error parsing API response: {str(e)}\nFull response: {result if 'result' in locals() else 'No result'}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def create_gradio_app():
    """
    Creates and returns the Gradio interface.
    """
    with gr.Blocks(title="Math Problem Explainer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔢 Math Problem Explainer")
        gr.Markdown("Enter a math problem and get a step-by-step explanation using AI!")
        
        with gr.Row():
            with gr.Column(scale=2):
                problem_input = gr.Textbox(
                    label="Math Problem",
                    placeholder="Enter your math problem here (e.g., 'Solve for x: 2x + 5 = 15')",
                    lines=3,
                    max_lines=5
                )
                
                explain_btn = gr.Button("Explain Problem", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                explanation_output = gr.Textbox(
                    label="Step-by-Step Explanation",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Example problems
        gr.Markdown("### 📚 Try these example problems:")
        examples = gr.Examples(
            examples=[
                ["Solve for x: 3x - 7 = 14"],
                ["Find the area of a circle with radius 5 cm"],
                ["What is the derivative of f(x) = x² + 3x - 2?"],
                ["Simplify: (2x + 3)(x - 4)"],
                ["Solve the quadratic equation: x² - 5x + 6 = 0"]
            ],
            inputs=[problem_input],
            cache_examples=False
        )
        
        # Event handlers
        explain_btn.click(
            fn=explain_math_problem,
            inputs=[problem_input],
            outputs=[explanation_output],
            show_progress=True
        )
        
        # Allow Enter key to submit
        problem_input.submit(
            fn=explain_math_problem,
            inputs=[problem_input],
            outputs=[explanation_output],
            show_progress=True
        )
    
    return app

def main():
    """
    Main function to launch the Gradio app.
    """
    print("🚀 Starting Math Problem Explainer...")
    
    app = create_gradio_app()
    
    # Launch the app
    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_api=True,
        share=False
    )

if __name__ == "__main__":
    main()
