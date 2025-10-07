# ALFA Project Restructuring - October 7, 2025

## 🎯 Overview

The ALFA project has been restructured from a single-app repository into a **monorepo** with separate applications for the Gradio interface and documentation site.

---

## 📁 New Project Structure

```
alfa/
├── apps/
│   ├── gradio/                  # Gradio web application
│   │   ├── main.py             # Main app entry point
│   │   ├── prompts/            # Subject-specific prompts
│   │   │   ├── video_scenario_math.txt
│   │   │   ├── video_scenario_chemistry.txt
│   │   │   ├── video_scenario_physics.txt
│   │   │   └── video_scenario_cs.txt
│   │   └── manim_scene.txt     # Manim generation prompt
│   │
│   └── docs/                   # Docusaurus documentation
│       ├── package.json
│       ├── docusaurus.config.js
│       ├── sidebars.js
│       ├── src/
│       │   ├── components/     # React components
│       │   │   ├── Hero.jsx    # Landing page hero with promo video
│       │   │   ├── Features.jsx
│       │   │   ├── Stats.jsx
│       │   │   └── SplitText.jsx  # React Bits animation
│       │   ├── css/
│       │   │   └── custom.css
│       │   └── pages/
│       │       └── index.jsx   # Landing page
│       ├── docs/               # Documentation content
│       │   ├── intro.md
│       │   ├── installation.md
│       │   ├── quick-start.md
│       │   └── ...
│       └── static/
│           ├── assets/
│           │   └── alfa.promo.v1.mp4  # Promo video
│           └── img/
│
├── .github/
│   └── workflows/
│       ├── deploy-docs.yml     # Docs deployment to GitHub Pages
│       └── ci.yml              # CI/CD testing
│
├── output/                     # Generated videos (unchanged)
├── assets/                     # Project assets (unchanged)
├── .credentials/               # API credentials (unchanged)
│
├── run-gradio.py               # Launcher for Gradio app
├── package.json                # Root npm scripts
├── pyproject.toml              # Python dependencies
├── Dockerfile                  # Docker image for Manim
├── docker-compose.yml          # Docker compose config
├── README.md                   # Updated main README
├── CONTRIBUTING.md             # Contribution guidelines
├── .env.example                # Environment variable template
└── RESTRUCTURING.md            # This file
```

---

## 🔄 What Changed

### 1. Gradio App → `apps/gradio/`

**Before:**
- `main.py` at root
- Prompts in `prompts/`

**After:**
- `apps/gradio/main.py`
- `apps/gradio/prompts/`

**Path Updates:**
- Credentials: Now use `Path(__file__).parent.parent.parent / ".credentials"`
- Output: Now use `Path(__file__).parent.parent.parent / "output"`
- All paths adjusted to work from `apps/gradio/` location

### 2. Documentation → `apps/docs/`

**New Docusaurus Site:**
- Modern React-based documentation
- Animated landing page with React Bits
- Embedded promo video (`alfa.promo.v1.mp4`)
- Subject-specific guides
- Examples and tutorials

**Features:**
- 🎬 Animated hero section with SplitText
- 📺 Promo video showcase
- 📊 Interactive stats section
- 🎨 Dark mode support
- 📱 Mobile responsive

### 3. GitHub Actions

**New Workflows:**

#### `deploy-docs.yml`
- Automatically builds and deploys docs to GitHub Pages
- Triggers on push to `main` branch
- Deploys to: `https://ozioisgood.github.io/alfa/`

#### `ci.yml`
- Tests documentation build
- Validates Gradio app syntax
- Tests Docker image build
- Runs on PRs and pushes

### 4. Root Files

**New Files:**
- `run-gradio.py` - Launch script for Gradio app
- `package.json` - Root npm scripts for convenience
- `CONTRIBUTING.md` - Contribution guidelines
- `.env.example` - Environment variable template
- `RESTRUCTURING.md` - This file

**Updated Files:**
- `README.md` - Complete rewrite with badges, better structure
- `.gitignore` - Added node_modules, build directories

---

## 🚀 How to Use

### Running Gradio App

```bash
# From root directory
uv run python run-gradio.py

# Or directly
cd apps/gradio && uv run python main.py
```

### Running Documentation Site

```bash
# Install dependencies (first time)
npm run docs:install

# Start development server
npm run docs

# Build for production
npm run docs:build

# Serve built site
npm run docs:serve
```

### Docker Commands

```bash
# Build Manim image
npm run docker:build

# Test Docker
npm run docker:test
```

---

## 🎨 Documentation Features

### Landing Page

The new landing page (`apps/docs/src/pages/index.jsx`) includes:

1. **Hero Section**
   - Animated title using SplitText from React Bits
   - Gradient background with pattern overlay
   - Subject badges (Math, Chemistry, Physics, CS)
   - CTA buttons (Get Started, GitHub)
   - **Promo video** (`assets/alfa.promo.v1.mp4`)

2. **Features Section**
   - 6 feature cards with icons
   - Scroll-triggered animations
   - Hover effects

3. **Stats Section**
   - "By the Numbers" showcase
   - Animated counters
   - Key metrics (4 subjects, 9 models, etc.)

### React Bits Integration

Custom `SplitText` component based on React Bits:
- Letter-by-letter animation
- Configurable timing and easing
- Framer Motion powered
- Smooth entrance effects

### Styling

Custom CSS (`apps/docs/src/css/custom.css`):
- Gradient backgrounds
- Animated elements
- Dark mode support
- Subject-specific color schemes
- Responsive design

---

## 📊 GitHub Pages Deployment

### Setup

1. Go to repository **Settings** → **Pages**
2. Set **Source** to "GitHub Actions"
3. Workflow will auto-deploy on push to `main`

### URL

Documentation will be live at:
```
https://ozioisgood.github.io/alfa/
```

### Build Process

1. Push changes to `apps/docs/` on `main` branch
2. GitHub Actions runs `deploy-docs.yml`
3. Builds Docusaurus site
4. Deploys to GitHub Pages
5. Live in ~2-5 minutes

---

## 🔧 Migration Notes

### For Contributors

**Before restructuring:**
```bash
python main.py
```

**After restructuring:**
```bash
uv run python run-gradio.py
# or
npm run gradio
```

### Path Changes in Code

If you have local modifications to `main.py`:

**Old:**
```python
credentials_path = Path(__file__).parent / ".credentials" / "alfa_gcp_sa.json"
```

**New:**
```python
credentials_path = Path(__file__).parent.parent.parent / ".credentials" / "alfa_gcp_sa.json"
```

### Environment Variables

Update your `.env` file based on `.env.example`:
- Copy `.env.example` to `.env`
- Fill in your API keys and credentials

---

## 🎯 Benefits of Restructuring

### 1. **Separation of Concerns**
- Gradio app is independent
- Documentation is separate
- Easier to maintain each

### 2. **Better Developer Experience**
- Clear project structure
- Easy to find files
- npm scripts for convenience

### 3. **Professional Documentation**
- Modern, animated landing page
- Comprehensive guides
- Automatic deployment

### 4. **CI/CD Pipeline**
- Automated testing
- Automatic docs deployment
- Quality assurance

### 5. **Scalability**
- Easy to add more apps
- Modular architecture
- Future-proof structure

---

## 📝 Next Steps

### For Project Maintainers

1. **Enable GitHub Pages**
   - Go to Settings → Pages
   - Enable GitHub Actions deployment

2. **Add Documentation**
   - Fill in subject guides (`docs/subjects/`)
   - Add more examples
   - Create tutorials

3. **Update Assets**
   - Add video poster image
   - Create favicon
   - Add logo SVG for docs navbar

### For Contributors

1. **Test the Changes**
   - Run Gradio app with new structure
   - Build documentation locally
   - Verify Docker still works

2. **Report Issues**
   - Open issues for any problems
   - Suggest improvements
   - Submit PRs

---

## 🐛 Known Issues

1. **First-time npm install**
   - Run `npm run docs:install` first
   - Then `npm run docs`

2. **Promo video path**
   - Video must be in `apps/docs/static/assets/`
   - Accessed via `/alfa/assets/alfa.promo.v1.mp4`

3. **Docker paths**
   - Volume mounts work from root directory
   - Ensure output/ is at root level

---

## 📞 Support

Questions about the restructuring?

- 📖 Read the updated [README.md](../README.md)
- 💬 Open an [issue](https://github.com/OZIOisgood/alfa/issues)
- 🔍 Check [documentation](https://ozioisgood.github.io/alfa/)

---

## ✅ Checklist for Fresh Clone

If you're cloning ALFA after restructuring:

- [ ] Clone repository
- [ ] Run `uv sync`
- [ ] Copy `.env.example` to `.env`
- [ ] Add your credentials to `.env`
- [ ] Place service account JSON in `.credentials/`
- [ ] Build Docker: `npm run docker:build`
- [ ] Test Gradio: `npm run gradio`
- [ ] (Optional) Test docs: `npm run docs:install && npm run docs`

---

**ALFA is now better organized and ready to scale!** 🚀
