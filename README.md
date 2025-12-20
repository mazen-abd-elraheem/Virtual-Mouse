# AI Virtual Mouse Controller - Two-Hand Edition v5.4

A professional-grade, real-time hand gesture-based mouse controller using computer vision and machine learning. This version separates control into two hands for maximum precision and usability:
- **Left Hand**: Selects the **Mode** (Scroll, Zoom, Volume, etc.)
- **Right Hand**: Performs the **Action** (Move, Click, Zoom In/Out, Volume Up/Down)

## Features

- **Dual Hand Control**: Innovative separation of concerns (Left: Context, Right: Action)
- **Discrete Gestures**: Reliable, simple gestures for actions instead of finicky continuous tracking
    - **Zoom**: 1 Finger for Zoom In, 2 Fingers for Zoom Out
    - **Scroll**: 1 Finger for Scroll Up, 2 Fingers for Scroll Down
    - **Volume**: 1 Finger for Volume Up, 2 Fingers for Volume Down
- **Robust Detection**: Relaxed constraints for comfortable use (e.g., Ring finger doesn't need to be perfectly hidden)
- **Real-time Tracking**: Uses MediaPipe for robust hand detection and landmark tracking
- **Smooth Cursor Movement**: Implements moving average smoothing to eliminate jitter
- **Active Region Mapping**: 
    - **Right Hand**: Mapped to the yellow box region (for cursor control)
    - **Left Hand**: Works anywhere in the full camera view
- **Visual Feedback**: Real-time display of active modes, gestures, and tracking areas

## gestures Guide

### LEFT HAND (The Activator)
The Left Hand determines **what mode** you are in. It works anywhere on the screen!

| Gesture | Fingers | Mode | Description |
| :--- | :--- | :--- | :--- |
| **Open Hand** | 🖐️ 5 Fingers | **SCROLL MODE** | Use Right Hand to scroll up/down |
| **Zoom Sign** | 🤘 Thumb + Pinky | **ZOOM MODE** | Use Right Hand to zoom in/out |
| **Phone Sign** | 🤙 Pinky Only | **VOLUME MODE** | Use Right Hand to adjust volume |
| **Peace Sign** | ✌️ Index + Middle | **MEDIA MODE** | Control music/video playback |
| **Rock Sign** | 🤟 Thumb+Index+Pinky | **WINDOW MODE** | Minimize/Maximize/Close windows |
| **Fist** | ✊ Closed Fist | **PRECISION MODE** | Extra smooth cursor movement |
| **None** | (Scanning) | **STANDARD MODE** | Normal mouse usage |

### RIGHT HAND (The Action)
The Right Hand performs the action based on the current mode.

#### 1. Standard Mode (No Special Left Hand Gesture)
- **Move Cursor**: Point with Index finger (inside yellow box)
- **Left Click**: ✌️ Two fingers (Index + Middle)
- **Right Click**: 🤟 Three fingers (Index + Middle + Ring)
- **Double Click**: 🖐️ Four fingers
- **Drag & Drop**: 🤏 Pinch Thumb & Index tightly

#### 2. Scroll / Zoom / Volume Modes
These modes now use the same simple **Discrete Gestures** for consistency:

| Action | Gesture | Scroll Mode | Zoom Mode | Volume Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Increase / Up** | 👆 **1 Finger** | Scroll Up | Zoom In | Volume Up |
| **Decrease / Down** | ✌️ **2 Fingers** | Scroll Down | Zoom Out | Volume Down |

#### 3. Media Mode
- **Play/Pause**: 👆 1 Finger
- **Next Track**: ✌️ 2 Fingers
- **Prev Track**: 🤟 3 Fingers

#### 4. Window Mode
- **Minimize**: 👆 1 Finger
- **Maximize**: ✌️ 2 Fingers
- **Close**: 🤟 3 Fingers
- **Screenshot**: 🖐️ 4 Fingers

## System Requirements

- Python 3.8 or higher
- Webcam
- Windows (Recommended for full feature support)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python ai_virtual_mouse.py
```

## Usage Tips

- **Right Hand Area**: Keep your Right Hand inside the **Yellow Box** to move the cursor. 
- **Left Hand Freedom**: Your Left Hand can be anywhere in the camera view to switch modes.
- **Lighting**: Ensure good lighting for best hand detection.
- **Distance**: Sit at a comfortable distance (approx. 50-80cm) from the camera.

## Troubleshooting

### Zoom Not Working?
- Ensure you are holding the **Zoom Gesture** (Thumb + Pinky) with your Left Hand first.
- Then use 1 Finger (Index) on Right Hand to Zoom In, or 2 Fingers to Zoom Out.

### Volume Not Working?
- Ensure you are holding the **Volume Gesture** (Pinky only) with your Left Hand.
- Note: We made this easier! You don't need to force your Ring finger down perfectly.

### Cursor Jittery?
- Try **Precision Mode** (Left Hand Fist) for smoother control.
- Adjust `smoothing_factor` in the specific class settings if needed.

## License

This code is provided as-is for educational and commercial use.

## Author

Senior Python Developer & Computer Vision Engineer
December 2024
