<div align="center"><div align="center"><div align="center">

  <img src="assets/alfa.png" alt="ALFA Logo" width="200" />

  	<img src="assets/alfa.png" alt="ALFA Logo" height="50%" />	<img src="assets/alfa.png" alt="ALFA Logo" height="50%" />

  # ALFA

  </div></div>

  **From problem to AI generated explainer animations**

  

  [![Documentation](https://img.shields.io/badge/docs-live-brightgreen)](https://ozioisgood.github.io/alfa/)

  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)# ALFA — From problem to AI generated explainer animations# ALFA — Short Math Explainer Generator

  [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

  

  [📖 Documentation](https://ozioisgood.github.io/alfa/) · [🚀 Quick Start](#quick-start) · [💡 Examples](#examples)

</div>> **TL;DR:** ALFA is an educational video generator that turns any problem or concept into a **30–180s** animated explanation with voice-over narration. Supports **Math, Chemistry, Physics, and Computer Science**!> **TL;DR:** ALFA is an app that turns any math problem you paste in into a **30–180s** video explanation with voice-over narration.



---



## 🎬 What is ALFA?## What it does## What it does



ALFA transforms any educational problem into a **beautiful animated video with AI narration** in minutes.



**Supported Subjects:*** You enter a problem or concept in any supported subject* You paste a math problem.

* 📐 **Mathematics** - Equations, graphs, geometry, calculus

* 🧪 **Chemistry** - Reactions, molecules, balancing, stoichiometry* ALFA creates a short, clear, narrated video that shows the **steps** and the **why*** ALFA creates a short, clear, narrated video that shows the **steps** and the **why**.

* ⚡ **Physics** - Forces, motion, energy, circuits

* 💻 **Computer Science** - Algorithms, data structures, Big-O* Generates professional animations with synchronized voice-overs



## ✨ Key Features## Who it’s for



* **🎙️ AI Voice Narration** - Natural speech synthesis via Google Gemini TTS## Supported Subjects

* **🎬 Professional Animations** - Manim-powered visuals (same engine as 3Blue1Brown)

* **🚀 Lightning Fast** - Generate 60-90s videos in just 5-10 minutesStudents, self‑learners, and busy folks who want fast, visual clarity instead of long lectures.

* **🎨 Quality Options** - 480p, 720p, 1080p, or 4K rendering

* **🤖 9 AI Models** - Choose from Gemini, GPT-4, Claude, and more* 📐 **Mathematics** - Equations, geometry, calculus, algebra, and more

* **🐳 Docker Support** - Isolated rendering environment with full LaTeX support

* 🧪 **Chemistry** - Reactions, molecular structures, stoichiometry, balancing equations## Why it’s useful

## 📺 Example Output

* ⚡ **Physics** - Kinematics, forces, energy, circuits, waves, and dynamics

**Input:** `Find the area of a circle with radius 5 cm`

* 💻 **Computer Science** - Algorithms, data structures, sorting, searching, complexity analysis* **One problem → one concise video**

**Output:** A 75-second video featuring:

- Animated circle drawing* **Visual-first** explanations that highlight the key idea

- Radius visualization  

- Formula derivation (A = πr²)## Who it's for* **Plain language** and friendly pacing

- Step-by-step calculation

- Final answer with narration



[🎥 Watch Demo Video](./assets/alfa.promo.v1.mp4)Students, self‑learners, educators, and anyone who wants fast, visual understanding instead of long lectures or dense textbooks.## How it works (simple)



## 🚀 Quick Start



### Prerequisites## Why it's useful1. **Input**: Paste your problem + choose quality & voice-over options



- Python 3.13+2. **Generate**: AI creates script, animations, and voice-over

- UV package manager

- Docker (recommended)* **One problem → one concise video**3. **Process**: Videos are rendered and combined automatically

- FFmpeg

* **Visual-first** explanations that highlight the key idea4. **Output**: A complete video ready to watch or download

### Installation

* **Subject-specific** visualizations (molecular models, force diagrams, code execution, graphs)

```bash

# Clone the repository* **Plain language** and friendly pacing> Built with **Gradio** (UI), **GPT-4** (script generation), **Manim** (animations), **Google TTS** (voice-over), and **FFmpeg** (video processing).

git clone https://github.com/OZIOisgood/alfa.git

cd alfa* **Professional quality** with smooth animations and clear audio



# Install dependencies## Quick Start

uv sync

## How it works (simple)

# Build Docker image (recommended)

docker build -t alfa-manim:latest .```bash



# Set up credentials (see docs)1. **Select Subject**: Choose Math, Chemistry, Physics, or Computer Science# Install dependencies

cp .env.example .env

# Add your API keys to .env2. **Input Problem**: Enter your problem or concept to explainuv sync



# Run the Gradio app3. **Generate**: AI creates subject-specific script, animations, and voice-over

uv run python run-gradio.py

```4. **Process**: Videos are rendered and combined automatically# Start the app



Visit `http://127.0.0.1:7865` to start creating videos!5. **Output**: A complete educational video ready to watch or shareuv run python main.py



📖 **Full installation guide:** [Documentation](https://ozioisgood.github.io/alfa/docs/installation)```



## 💡 Examples> Built with **Gradio** (UI), **Gemini/GPT-4** (script generation), **Manim** (animations), **Google TTS** (voice-over), and **FFmpeg** (video processing).



### MathematicsVisit **http://127.0.0.1:7865** and try it out!

```

Problem: Find the slope between points (2, 4) and (6, 12)## Quick Start

→ Coordinate grid, point plotting, slope calculation

```📖 See [QUICK_START.md](QUICK_START.md) for detailed usage guide.



### Chemistry```bash

```

Problem: Balance: CH₄ + O₂ → CO₂ + H₂O# Install dependencies## Status

→ Molecular models, atom counting, balanced equation

```uv sync



### Physics* ✅ Core features complete: video generation, audio sync, quality selection

```

Problem: Ball dropped from 45m, find time to hit ground# Start the app* ✅ Docker support for isolated Manim rendering

→ Free-body diagram, kinematics equations, solution

```uv run python main.py* ✅ Comprehensive LaTeX package support



### Computer Science```* 🚧 Testing and optimization ongoing

```

Problem: Explain binary search on [3, 7, 12, 18, 25, 31, 42]* 🎬 Sample videos: Coming soon

→ Array visualization, pointers, O(log n) complexity

```Visit **http://127.0.0.1:7865** and try it out!



## 📁 Project Structure



```📖 See [QUICK_START.md](QUICK_START.md) for detailed usage guide.## What's next

alfa/

├── apps/

│   ├── gradio/          # Gradio web application

│   │   ├── main.py      # Main app entry point## Example Use Cases* 📹 Publish demo videos and examples

│   │   └── prompts/     # Subject-specific prompts

│   └── docs/            # Docusaurus documentation site* 🌐 Deploy web version

├── output/              # Generated videos

├── assets/              # Static assets### Mathematics

├── .credentials/        # API credentials (not in git)

└── run-gradio.py        # Launch script```## Want to help?

```

"Find the slope of the line passing through points (2, 4) and (6, 12)"

## 🎓 Documentation

→ Coordinate grid, point plotting, slope calculation with visual formulaOpen an issue with a problem you’d like explained in under two minutes.

Our comprehensive documentation includes:

```

- **Getting Started** - Installation, configuration, first video

- **Subject Guides** - Best practices for each subject area---

- **Advanced Topics** - Custom prompts, Docker, API usage

- **Examples** - Sample problems and outputs### Chemistry



**📖 Visit:** [ozioisgood.github.io/alfa](https://ozioisgood.github.io/alfa/)```**ALFA** — *Paste a problem. Get a tiny, tidy video.*



## 🛠️ Technology Stack"Balance the combustion reaction: CH₄ + O₂ → CO₂ + H₂O"

→ Molecular structures, atom counting, balanced equation with coefficients

- **Frontend:** Gradio for web UI```

- **AI/LLM:** Vertex AI (Gemini), OpenRouter (GPT-4, Claude)

- **Animations:** Manim Community Edition### Physics

- **Voice:** Google Cloud Text-to-Speech```

- **Video:** FFmpeg for processing"A ball is thrown horizontally from a 20m cliff at 15 m/s. Find time to impact."

- **Containerization:** Docker for rendering→ Free body diagram, trajectory path, kinematic equations solved step-by-step

- **Documentation:** Docusaurus with React```



## 📊 Performance### Computer Science

```

| Quality | Resolution | Render Time* | File Size** |"Explain binary search algorithm with array [5, 12, 23, 42, 57, 68, 91]"

|---------|-----------|--------------|-------------|→ Array visualization, pointer movement, comparisons, O(log n) complexity

| Low     | 480p      | 2-3 min      | ~10 MB      |```

| Medium  | 720p      | 5-10 min     | ~30 MB      |

| High    | 1080p     | 15-25 min    | ~150 MB     |## Features

| 4K      | 2160p     | 30-60 min    | ~500 MB     |

* ✅ **Multi-subject support**: Math, Chemistry, Physics, Computer Science

*For typical 60-90s videos  * ✅ **9 LLM models**: Gemini (Vertex AI) + GPT-4, Claude (OpenRouter)

**Approximate, varies by content* ✅ **Subject-specific prompts**: Optimized for each discipline

* ✅ **Professional animations**: Manim-powered visualizations

## 🤝 Contributing* ✅ **Voice synthesis**: Google TTS with natural narration

* ✅ **Quality options**: 480p, 720p, 1080p, 4K

We welcome contributions! Areas where you can help:* ✅ **Docker support**: Isolated rendering environment

* ✅ **Section limiting**: Test with fewer sections

- 🧪 Test with different problem types* ✅ **Sequential generation**: Reliable scene-by-scene processing

- 📝 Improve documentation

- 🎨 Enhance prompts for better outputs## Status

- 🐛 Report bugs and issues

- ✨ Suggest new features* ✅ Core features complete: multi-subject video generation, audio sync, quality selection

* ✅ Subject-specific prompts for Math, Chemistry, Physics, CS

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.* ✅ Docker support for isolated Manim rendering

* ✅ Comprehensive LaTeX package support

## 📄 License* 🚧 Testing and optimization ongoing

* 🎬 Sample videos: Coming soon

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## What's next

## 🙏 Acknowledgments

* 📹 Publish demo videos for each subject

- **Manim Community** - For the amazing animation engine* 🌐 Deploy web version

- **3Blue1Brown** - Inspiration for visual math explanations* 📚 Add more subject areas (Biology, Economics, etc.)

- **Google Cloud** - Text-to-Speech and Vertex AI* 🎨 Enhanced visualization templates

- **OpenAI, Anthropic** - LLM capabilities

## Want to help?

## 📞 Support

Open an issue with a problem from any subject you'd like explained in under two minutes.

- 📖 [Documentation](https://ozioisgood.github.io/alfa/)

- 💬 [GitHub Issues](https://github.com/OZIOisgood/alfa/issues)---

- ⭐ [Star on GitHub](https://github.com/OZIOisgood/alfa)

**ALFA** — *From problem to AI generated explainer animations*  

---Paste a problem. Get a visual explanation.


<div align="center">
  <strong>ALFA</strong> - <em>Transform problems into visual understanding</em>
  
  Made with ❤️ for educators and learners worldwide
</div>
