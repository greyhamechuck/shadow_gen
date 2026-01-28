# Realistic Shadow Generator

A powerful AI-driven tool for generating realistic shadows for compositing subjects into backgrounds. This application uses depth estimation and ray marching to create shadows that interact realistically with the environment, such as bending over uneven surfaces or climbing walls.

## Features

- **Automatic Background Removal**: Seamlessly extracts the subject from foreground images using `rembg`.
- **AI Depth Estimation**: Utilizes the `Depth Anything` model to understand the 3D structure of your background scene.
- **Realistic Shadow Rendering**:
  - **Ray Marching**: Shadows physically interact with the background geometry (e.g., distorting over obstacles).
  - **Contact Hardening**: Shadows are sharp near the contact point and soften as they extend outward, mimicking real-world physics.
  - **Directional Control**: Adjust light angle and elevation to match the scene's lighting.
- **Interactive GUI**: Built with PyQt6, featuring real-time sliders for:
  - Light Angle & Elevation
  - Shadow Softness & Opacity
  - Ray Distance (Depth distortion strength)
  - Subject Vertical Position
- **Layer Export**: Save not just the composite, but also isolated shadow layers, masks, and depth maps for further editing in Photoshop or other tools.

## Requirements

- Python 3.8+
- macOS (Optimized for Apple Silicon 'mps' acceleration), Windows, or Linux

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd shadow_gen
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    python shadow_create.py
    ```

2.  **Workflow**:
    - Click **"1. Load Subject"** to choose your foreground image (e.g., a person or object). The app will automatically remove the background.
    - Click **"2. Load Background"** to choose the environment image.
    - Click **"✨ AI Auto-Depth"** to generate a depth map of the background. *This is crucial for the "wall climbing" shadow effects.*
    - Adjust the sliders (Angle, Elevation, Softness, etc.) until the shadow matches the scene.
    - Click **"💾 Save Layers"** to export the results.

## Controls

- **Angle**: Controls the horizontal direction of the light source (0-360°).
- **Elevation**: Controls how high the light source is (10-80°). Lower values create longer shadows.
- **Softness**: Controls the blur radius of the shadow.
- **Opacity**: Controls the transparency of the shadow.
- **Ray Distance**: Controls how strongly the shadow reacts to the background depth (e.g., how much it "climbs" a wall).
- **Position Y%**: Adjusts the vertical position of the subject in the scene.

## Output

When you click "Save Layers", the following files are continuously generated in the project directory:
- `composite.png`: The final merged image.
- `shadow_only.png`: The isolated shadow layer.
- `mask_debug.png`: The subject's cutout mask.
- `depth_map.png`: The computed depth map (if AI Auto-Depth was run).

## Technologies Used

- **Python**
- **PyQt6** for the user interface.
- **OpenCV & NumPy** for image processing and ray marching logic.
- **Hugging Face Transformers** for the `Depth Anything` model.
- **Rembg** for background removal.
- **PyTorch** for deep learning acceleration.
