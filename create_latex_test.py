"""
Test script to verify comprehensive LaTeX package availability in Docker.
"""
from pathlib import Path

# Create test directory
test_dir = Path("output/latex_test")
test_dir.mkdir(parents=True, exist_ok=True)

# Create comprehensive LaTeX test scene
test_script = test_dir / "LaTeXPackageTest.py"
test_script.write_text('''from manim import *

class LaTeXPackageTest(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"
        
        title = Text("LaTeX Package Test", font_size=48, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Test 1: amsmath - Basic math
        test1 = MathTex(
            r"\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
            font_size=36,
            color=WHITE
        )
        test1.shift(UP * 2)
        self.play(FadeIn(test1), run_time=1)
        
        # Test 2: amssymb & doublestroke - Number sets
        test2 = MathTex(
            r"f: \\mathbb{R} \\to \\mathbb{C}",
            font_size=36,
            color=BLUE
        )
        test2.shift(UP * 0.5)
        self.play(FadeIn(test2), run_time=1)
        
        # Test 3: physics - Bra-ket notation
        test3 = MathTex(
            r"\\langle \\psi | \\hat{H} | \\psi \\rangle = E",
            font_size=36,
            color=GREEN
        )
        test3.shift(DOWN * 1)
        self.play(FadeIn(test3), run_time=1)
        
        # Test 4: xcolor - Colored math
        test4 = MathTex(
            r"{\\color{red} x^2} + {\\color{blue} y^2} = {\\color{green} r^2}",
            font_size=36
        )
        test4.shift(DOWN * 2.5)
        self.play(FadeIn(test4), run_time=1)
        
        self.wait(2)
        
        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(test1),
            FadeOut(test2),
            FadeOut(test3),
            FadeOut(test4),
            run_time=1
        )
''', encoding='utf-8')

print("=" * 60)
print("LaTeX Package Test Script Created")
print("=" * 60)
print(f"📄 Script: {test_script}")
print()
print("This script tests:")
print("  ✓ amsmath - Integrals, fractions, sqrt")
print("  ✓ amssymb - Mathematical symbols")
print("  ✓ doublestroke - Number sets (ℝ, ℂ)")
print("  ✓ physics - Bra-ket notation")
print("  ✓ xcolor - Colored mathematics")
print()
print("To run after Docker build completes:")
print()
print("  docker run --rm \\")
print(f'    -v "${{PWD}}:/manim" \\')
print("    -w /manim \\")
print("    alfa-manim:latest \\")
print("    -qm \\")
print("    --media_dir /manim/output/latex_test/media \\")
print("    /manim/output/latex_test/LaTeXPackageTest.py \\")
print("    LaTeXPackageTest")
print()
print("=" * 60)
