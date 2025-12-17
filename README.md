# AI Virtual Mouse Controller

A professional-grade, real-time hand gesture-based mouse controller using computer vision and machine learning.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe for robust hand detection and landmark tracking
- **Smooth Cursor Movement**: Implements moving average smoothing to eliminate jitter
- **Gesture Recognition**: 
  - Move cursor with index finger
  - Left click by bringing index and middle fingers together
  - Right click with three fingers up (index, middle, ring)
- **Active Region Mapping**: Efficient coordinate mapping from camera view to full screen
- **Click Debouncing**: Prevents accidental multiple clicks
- **Visual Feedback**: Real-time display of hand landmarks, active region, FPS, and gesture states
- **Mirror Mode**: Natural mirrored camera feed (move right to go right)

## System Requirements

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

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

## Usage

### Starting the Application

```bash
python ai_virtual_mouse.py
```

### Hand Gestures

1. **Move Cursor**: 
   - Raise your index finger only
   - Keep middle finger down
   - Move within the green active region box

2. **Left Click**: 
   - Raise both index and middle fingers
   - Bring the fingertips close together (pinch gesture)
   - A red line will appear when click is detected

3. **Right Click**: 
   - Raise index, middle, AND ring fingers simultaneously
   - This creates a distinct three-finger gesture

### Exiting
- Press 'q' while the camera window is active

## Configuration

You can customize the behavior by modifying parameters in the `VirtualMouse` class initialization:

```python
virtual_mouse = VirtualMouse(
    cam_width=640,              # Camera resolution width
    cam_height=480,             # Camera resolution height
    frame_reduction=100,        # Active region padding (pixels)
    smoothing_factor=5,         # Smoothing buffer size (higher = smoother but slower)
    click_threshold=30,         # Distance for click detection (pixels)
    debounce_time=0.3,          # Minimum time between clicks (seconds)
    dominant_hand='Right'       # 'Right' or 'Left'
)
```

## Technical Details

### Coordinate Mapping Logic

The system uses linear interpolation to map camera coordinates to screen coordinates:

```
screen_x = (cam_x - active_min) / (active_max - active_min) * screen_width
```

This creates a proportional mapping where:
- The edges of the active region map to the edges of the screen
- Small hand movements translate to full cursor range
- Users don't need to stretch to reach screen edges

### Smoothing Algorithm

A moving average filter is applied to cursor coordinates:

```python
smooth_x = mean(last_N_x_coordinates)
smooth_y = mean(last_N_y_coordinates)
```

This eliminates camera shake and hand tremor while maintaining responsiveness.

### Click Debouncing

Clicks are debounced using time-based filtering:
- A minimum time interval (default 0.3s) must pass between clicks
- Prevents gesture recognition noise from causing multiple clicks

## Performance Optimization

The system is optimized for low latency:
- PyAutoGUI failsafe disabled for faster response
- Single hand tracking to reduce computational load
- Efficient coordinate smoothing with deque data structure
- Direct pixel manipulation without unnecessary processing

## Troubleshooting

### Camera Not Opening
- Check if another application is using the webcam
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Cursor Not Moving Smoothly
- Increase `smoothing_factor` for more smoothing (5-10 recommended)
- Ensure good lighting conditions for better hand detection
- Keep hand within the green active region box

### Clicks Not Registering
- Adjust `click_threshold` (20-40 recommended)
- Ensure fingers are clearly visible to the camera
- Check that `debounce_time` is not too high

### Low FPS
- Reduce camera resolution
- Close other applications
- Ensure GPU drivers are updated (for MediaPipe acceleration)

## Notes on Gesture Design

The right-click gesture uses three fingers (index, middle, ring) rather than thumb pinch because:
- More reliable detection with MediaPipe
- More ergonomic and comfortable to perform repeatedly
- Clear visual distinction from left-click gesture
- Less prone to false positives

## Library Selection: PyAutoGUI vs AutoPy

This implementation uses **PyAutoGUI** because:
- Cross-platform compatibility (Windows, macOS, Linux)
- Active maintenance and community support
- Built-in safety features
- Extensive documentation

While AutoPy may have slightly lower latency on some systems, PyAutoGUI provides better reliability and compatibility for production use.

## License

This code is provided as-is for educational and commercial use.

## Author

Senior Python Developer & Computer Vision Engineer
December 2024
