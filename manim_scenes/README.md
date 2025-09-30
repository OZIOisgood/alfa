# Manim Scenes Directory

This directory contains auto-generated Manim scene files for video animations.

## Structure

Each file corresponds to one keyframe from the scenario JSON:
- Filename format: `{SectionName}_{KeyframeName}.py`
- Each file contains a single Scene class with the same name
- Animation duration matches or exceeds the voice-over length

## Usage

### Local Rendering (if Manim installed)
```bash
manim -pql SectionName_KeyframeName.py SectionName_KeyframeName
```

### Docker Rendering (recommended)
```bash
docker run --rm -v "%cd%:/manim" manimcommunity/manim manim -qm /manim/SectionName_KeyframeName.py SectionName_KeyframeName
```

### Batch Rendering All Scenes
```powershell
# PowerShell script to render all scenes
Get-ChildItem -Filter "*.py" | ForEach-Object {
    $className = $_.BaseName
    Write-Host "Rendering $className..."
    docker run --rm -v "${PWD}:/manim" manimcommunity/manim manim -qm "/manim/$($_.Name)" $className
}
```

## Quality Settings
- `-ql`: Low quality (480p, fast preview)
- `-qm`: Medium quality (720p)
- `-qh`: High quality (1080p)
- `-qk`: 4K quality (2160p)

## Output
Videos are saved to `media/videos/{SceneName}/{quality}/`

## Notes
- Each scene is self-contained and can be rendered independently
- Scenes are generated to match voice-over timing
- Edit the generated files to customize animations before rendering
