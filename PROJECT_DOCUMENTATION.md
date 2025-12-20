# AI Virtual Mouse - Technical Documentation

## 1. Project Overview
The **AI Virtual Mouse** is a sophisticated Human-Computer Interaction (HCI) system that enables users to control their computer using hand gestures captured by a standard webcam. Unlike simple tracking demos, this project implements a **Two-Hand Architecture** where the left hand manages "Context/Mode" and the right hand performs "Actions," mimicking complex keyboard+mouse interactions (like Ctrl+Zoom or Volume Keys).

## 2. Technology Stack
- **Language**: Python 3.8+
- **Computer Vision**: OpenCV (Frame capture, image processing, visualization)
- **ML/Tracking**: MediaPipe Hands (Google's ML solution for high-fidelity hand tracking)
- **Automation**: PyAutoGUI (Simulating mouse events and keyboard presses)
- **Math/Logic**: NumPy (Coordinate interpolation, distance calculations)

## 3. System Architecture

The system operates in a continuous loop with the following pipeline:

1.  **Input Capture**: Webcam captures frames at 30+ FPS.
2.  **Preprocessing**: Frames are flipped (mirror effect) and converted to RGB.
3.  **Hand Tracking (MediaPipe)**: 
    - Detects up to 2 hands.
    - Extracts 21 3D landmarks per hand.
    - Classifies 'Left' vs 'Right' hand.
4.  **Logic Layer**:
    - **Step A: Left Hand Analysis (Mode Detection)**: Analyzes finger states (Up/Down) to determine the *active mode* (e.g., Scroll, Zoom, Volume).
    - **Step B: Right Hand Analysis (Gesture Recognition)**: Analyzes the Right Hand's finger configuration *in the context of* the Left Hand's mode.
5.  **Action Execution**: Triggers OS-level events (Move Mouse, Scroll, Click) via PyAutoGUI.
6.  **Visual Feedback**: Renders overlays, modes, and landmarks on the display frame.

## 4. Key Algorithms

### 4.1 Hand Mode Detection
The system uses boolean logic based on finger states (Up/Down) to classify modes.
*Example Logic (Zoom Mode)*:
```python
if Thumb_Up and Pinky_Up and (Index, Middle, Ring are Down or Relaxed):
    Mode = ZOOM
```
*Innovation*: "Relaxed Constraints" were implemented to solve anatomical difficulties (e.g., lifting the Pinky naturally pulls the Ring finger). The algorithm tolerates the Ring finger state in specific modes to improve robustness.

### 4.2 Discrete vs Continuous Actions
The system employs two types of interaction models:
1.  **Continuous (Analog)**:
    -   *Cursor Movement*: Mapped directly from hand coordinates.
    -   *Smoothing*: Uses a `deque` buffer to average the last `N` positions, reducing jitter caused by webcam noise or hand tremors.
2.  **Discrete (Digital)**:
    -   *Volume/Zoom/Scroll*: Instead of tracking hand height (which is finicky), we uses distinct "Triggers".
    -   *1 Finger Up* -> Trigger "Up/Increase" signal.
    -   *2 Fingers Up* -> Trigger "Down/Decrease" signal.
    -   *Debouncing*: A `cooldown` timer prevents a single gesture from triggering an action hundreds of times per second.

### 4.3 Coordinate Mapping
To reach the corners of the screen without stretching the arm, the system maps a smaller "Active Region" (Yellow Box) in the camera frame to the full Screen Resolution.
```python
screen_x = interp(cam_x, (padding, cam_w - padding), (0, screen_w))
screen_y = interp(cam_y, (padding, cam_h - padding), (0, screen_h))
```

## 5. Class Structure

### `TwoHandVirtualMouse` (Main Class)
-   `__init__`: Sets up camera, constants, and deque buffers.
-   `run()`: The main event loop.
-   `detect_left_hand_mode()`: classification logic for the Context hand.
-   `detect_right_hand_gesture()`: classification logic for the Action hand.
-   `execute_* methods`: Handlers for specific actions (Volume, Zoom, etc.).
-   `smooth_coordinates()`: Moving average filter implementation.

### Enums
-   `HandGesture`: Defines all recognized right-hand states (MOVE, CLICK, ZOOM_IN, etc.).
-   `ActivationMode`: Defines all left-hand contexts (SCROLL, ZOOM, VOLUME, etc.).

## 6. Future Improvements
-   **Dynamic Thresholding**: Adjust click distance thresholds based on hand distance from camera (Z-depth).
-   **Custom Gesture Recording**: Allow users to define their own hand signs for shortcuts.
-   **Multi-Monitor Support**: Better mapping for dual-screen setups.
