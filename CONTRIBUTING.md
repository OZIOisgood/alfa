# Contributing to ALFA

Thank you for your interest in contributing to ALFA! 🎉

## Ways to Contribute

### 🐛 Reporting Bugs

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version, Docker version)
- Screenshots/videos if applicable

### 💡 Suggesting Features

Have an idea? Open an issue with:
- Clear description of the feature
- Use case / problem it solves
- Potential implementation approach (optional)

### 📝 Improving Documentation

- Fix typos or unclear explanations
- Add examples or tutorials
- Translate documentation

### 🧪 Testing

- Test with different problem types
- Report edge cases
- Verify installations on different platforms

### 💻 Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/alfa.git
cd alfa

# Install dependencies
uv sync

# Build Docker image
docker build -t alfa-manim:latest .

# Run tests (if available)
uv run pytest
```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## Pull Request Guidelines

- Link related issues
- Describe your changes clearly
- Update documentation if needed
- Ensure all tests pass
- Keep PRs focused (one feature/fix per PR)

## Questions?

Open an issue or discussion if you need help!

---

**Thank you for making ALFA better!** ❤️
