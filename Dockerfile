# Use the official Manim Community image as base
FROM manimcommunity/manim:v0.18.1

# Install additional LaTeX packages for better text rendering
# The base image already has basic LaTeX, but we add more comprehensive support
USER root

# Update package lists and install additional LaTeX packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-latex-recommended \
    texlive-science \
    texlive-fonts-recommended \
    cm-super \
    dvipng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER manimuser

# Set working directory
WORKDIR /manim

# The container will run manim commands passed as arguments
ENTRYPOINT ["manim"]
