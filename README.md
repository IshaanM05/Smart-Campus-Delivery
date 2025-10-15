# High-Performance Campus GPS Tracking Simulator

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. For GPU acceleration: Install CUDA toolkit and ensure Numba CUDA is configured.
3. Run: `python main.py`

## Features
- See feature summary in the user query.
- Interactive GUI: Use sliders to adjust agent count (1-1000+), FPS (15-60), etc.
- Click on simulation tab to add/remove landmarks (updates graph dynamically).

## Notes
- Animation may be slow for 1000+ agents; use LOD (auto-enabled).
- For production, deploy with commercial tools.