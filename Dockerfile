# Use the official Manim Community image as base
FROM manimcommunity/manim:v0.18.1

# Install additional LaTeX packages for comprehensive text rendering
# The base image already has basic LaTeX, but we add more comprehensive support
USER root

# Update package lists and install comprehensive LaTeX packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core LaTeX distributions
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-latex-recommended \
    texlive-science \
    texlive-fonts-recommended \
    texlive-lang-english \
    texlive-xetex \
    # Font packages
    cm-super \
    fonts-freefont-ttf \
    lmodern \
    # Additional utilities
    dvipng \
    dvisvgm \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note: Most LaTeX packages (amsmath, xcolor, physics, etc.) are already included
# in texlive-latex-extra, texlive-science, and texlive-fonts-extra.
# No need for tlmgr installation which can fail due to version mismatches.

# Switch back to non-root user
USER manimuser

# Set working directory
WORKDIR /manim

# The container will run manim commands passed as arguments
ENTRYPOINT ["manim"]

