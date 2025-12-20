"""
AI Virtual Mouse Controller - Two-Hand Edition v5.4 (DISCRETE GESTURES)
A real-time dual-hand gesture-based mouse controller using computer vision.

Author: Senior Python Developer & CV Engineer
Date: December 2024

UPDATES in v5.4:
- SCROLL: 1 finger = scroll UP, 2 fingers = scroll DOWN
- ZOOM: 1 finger = zoom IN, 2 fingers = zoom OUT
- All special modes use same simple gesture pattern
- Improved responsiveness
- Clean discrete gestures - no tracking needed
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque
from enum import Enum


class HandGesture(Enum):
    """Enumeration for different hand gestures"""
    NONE = 0
    MOVE = 1
    LEFT_CLICK = 2
    RIGHT_CLICK = 3
    DRAG = 4
    SCROLL = 5
    ZOOM_IN = 6
    ZOOM_OUT = 7
    VOLUME_UP = 8
    VOLUME_DOWN = 9
    DOUBLE_CLICK = 10
    MIDDLE_CLICK = 11
    SCREENSHOT = 12
    MEDIA_PLAY_PAUSE = 13
    MEDIA_NEXT = 14
    MEDIA_PREV = 15
    BRIGHTNESS_UP = 16
    BRIGHTNESS_DOWN = 17
    MINIMIZE_WINDOW = 18
    MAXIMIZE_WINDOW = 19
    CLOSE_WINDOW = 20
    TAB_NEXT = 21
    TAB_PREV = 22
    COPY = 23
    PASTE = 24
    UNDO = 25
    SELECT_ALL = 26


class ActivationMode(Enum):
    """Left hand activation modes"""
    NONE = 0
    SCROLL = 1      # 🖐️ All 5 fingers up
    ZOOM = 2        # 🤘 Thumb + Pinky only
    VOLUME = 3      # 🤙 Pinky only
    PRECISION = 4   # ✊ Fist (all down)
    MEDIA = 5       # ✌️ Peace sign (Index + Middle)
    WINDOW = 6      # 🤟 Rock sign (Thumb + Index + Pinky)
    SHORTCUT = 7    # 👌 OK sign (Thumb + Index touching)


class TwoHandVirtualMouse:
    """
    AI-powered two-hand virtual mouse controller v5.1.
    
    ══════════════════════════════════════════════════════════════════
    LEFT HAND (Activator) - Controls which mode is active:
    ══════════════════════════════════════════════════════════════════
    🖐️  All 5 fingers up        → SCROLL MODE (natural drag-style)
    🤘  Thumb + Pinky only      → ZOOM MODE
    🤙  Pinky only              → VOLUME MODE
    ✊  Fist (all down)         → PRECISION MODE
    ✌️  Index + Middle only     → MEDIA MODE (play/pause, next, prev)
    🤟  Thumb + Index + Pinky   → WINDOW MODE (minimize, maximize, close)
    👌  Thumb + Index pinch     → SHORTCUT MODE (copy, paste, undo)
    
    ══════════════════════════════════════════════════════════════════
    RIGHT HAND (Action) - Performs the action:
    ══════════════════════════════════════════════════════════════════
    
    STANDARD MODE (no left hand or PRECISION):
    👆  1 finger (index)             → MOVE cursor
    ✌️  2 fingers (index + middle)   → LEFT CLICK
    🤟  3 fingers (index+mid+ring)   → RIGHT CLICK
    🖐️  4 fingers                    → DOUBLE CLICK
    🤏  Tight pinch (thumb + index)  → DRAG
    👍  Thumb only                   → MIDDLE CLICK
    
    SCROLL MODE (left hand open):
    👆  1 finger (index only)        → SCROLL UP
    ✌️  2 fingers (index + middle)   → SCROLL DOWN
    
    ZOOM MODE (left hand thumb+pinky):
    👆  1 finger (index only)        → ZOOM IN
    ✌️  2 fingers (index + middle)   → ZOOM OUT
    
    VOLUME MODE (left hand pinky only):
    👆  1 finger (index)             → VOLUME UP
    ✌️  2 fingers (index + middle)   → VOLUME DOWN
    
    MEDIA MODE (left hand peace sign):
    👆  1 finger                     → PLAY/PAUSE
    ✌️  2 fingers                    → NEXT TRACK
    🤟  3 fingers                    → PREVIOUS TRACK
    
    WINDOW MODE (left hand rock sign):
    👆  1 finger                     → MINIMIZE WINDOW
    ✌️  2 fingers                    → MAXIMIZE WINDOW
    🤟  3 fingers                    → CLOSE WINDOW
    🖐️  4 fingers                    → SCREENSHOT
    
    SHORTCUT MODE (left hand OK gesture):
    👆  1 finger                     → COPY (Ctrl+C)
    ✌️  2 fingers                    → PASTE (Ctrl+V)
    🤟  3 fingers                    → UNDO (Ctrl+Z)
    🖐️  4 fingers                    → SELECT ALL (Ctrl+A)
    ══════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, 
                 cam_width=640, 
                 cam_height=480,
                 frame_reduction=100,
                 smoothing_factor=7,
                 click_threshold=30,
                 debounce_time=0.3):
        """
        Initialize the Two-Hand Virtual Mouse system.
        
        Args:
            cam_width: Camera capture width
            cam_height: Camera capture height
            frame_reduction: Pixels to reduce from frame edges
            smoothing_factor: Number of previous coordinates to average
            click_threshold: Distance threshold for click detection
            debounce_time: Minimum time between clicks
        """
        # Camera settings
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Active region settings
        self.frame_reduction = frame_reduction
        self.active_x_min = frame_reduction
        self.active_x_max = cam_width - frame_reduction
        self.active_y_min = frame_reduction
        self.active_y_max = cam_height - frame_reduction
        
        # Smoothing buffer
        self.smoothing_factor = smoothing_factor
        self.prev_x_coords = deque(maxlen=smoothing_factor)
        self.prev_y_coords = deque(maxlen=smoothing_factor)
        
        # Enhanced smoothing for edges
        self.edge_smoothing_factor = smoothing_factor + 3
        self.edge_threshold = 50
        
        # Click detection
        self.click_threshold = 40
        self.debounce_time = 0.4
        self.last_click_time = 0
        self.last_gesture = HandGesture.NONE
        
        # Drag state
        self.is_dragging = False
        self.drag_start_pos = None
        self.drag_pinch_threshold = 25
        
        # Natural scroll state (discrete gestures)
        self.last_scroll_time = 0
        
        # Zoom state (discrete gestures)
        self.last_zoom_action_time = 0
        self.zoom_mode_active = False
        
        # Volume state
        self.volume_sensitivity = 30  # Pixels of movement for volume change
        self.last_volume_y = None
        self.volume_mode_active = False
        self.volume_cooldown = 0.2
        self.last_volume_time = 0
        
        # Precision mode state
        self.precision_mode_active = False
        self.precision_smoothing_factor = 15
        
        # Media mode state
        self.media_mode_active = False
        self.last_media_action_time = 0
        self.media_debounce = 0.5
        
        # Window mode state
        self.window_mode_active = False
        self.last_window_action_time = 0
        self.window_debounce = 0.5
        
        # Shortcut mode state
        self.shortcut_mode_active = False
        self.last_shortcut_action_time = 0
        self.shortcut_debounce = 0.3
        
        # Current activation mode
        self.current_mode = ActivationMode.NONE
        
        # Hand tracking state
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.left_hand_detected = False
        self.right_hand_detected = False
        
        # Store right hand info for delayed processing
        self.stored_right_fingers = None
        self.stored_right_landmarks = None
        
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # FPS calculation
        self.prev_time = 0
        self.curr_time = 0
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        # Visual feedback colors (BGR)
        self.COLOR_LEFT_HAND = (0, 255, 0)      # Green
        self.COLOR_RIGHT_HAND = (255, 0, 0)     # Blue
        self.COLOR_ACTIVE_FRAME = (0, 255, 255) # Yellow
        self.COLOR_INDEX_TIP = (255, 0, 255)    # Magenta
        self.COLOR_TEXT = (255, 255, 255)       # White
        self.COLOR_SCROLL_MODE = (0, 255, 255)  # Cyan
        self.COLOR_ZOOM_MODE = (255, 165, 0)    # Orange
        self.COLOR_VOLUME_MODE = (255, 0, 255)  # Magenta
        self.COLOR_PRECISION = (0, 165, 255)    # Orange-Red
    
    def get_finger_status(self, landmarks, hand_label):
        """
        Determine which fingers are up or down with improved accuracy.
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_label: 'Left' or 'Right'
            
        Returns:
            List of 5 boolean values [Thumb, Index, Middle, Ring, Pinky]
        """
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        
        # Thumb (special case - check horizontal distance)
        if hand_label == 'Right':
            # For right hand: thumb tip should be to the right of thumb base
            fingers.append(landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 2].x)
        else:
            # For left hand: thumb tip should be to the left of thumb base
            fingers.append(landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 2].x)
        
        # Other fingers - compare tip with PIP joint (more reliable than DIP)
        for id in range(1, 5):
            # Finger is up if tip is significantly higher than PIP joint
            is_up = landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y - 0.02
            fingers.append(is_up)
        
        return fingers
    
    def get_landmark_position(self, landmarks, landmark_id):
        """Get pixel coordinates of a specific landmark."""
        landmark = landmarks[landmark_id]
        x = int(landmark.x * self.cam_width)
        y = int(landmark.y * self.cam_height)
        return x, y
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def map_to_screen(self, cam_x, cam_y):
        """Map camera coordinates to screen coordinates."""
        screen_x = np.interp(
            cam_x,
            (self.active_x_min, self.active_x_max),
            (0, self.screen_width)
        )
        screen_y = np.interp(
            cam_y,
            (self.active_y_min, self.active_y_max),
            (0, self.screen_height)
        )
        
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return int(screen_x), int(screen_y)
    
    def smooth_coordinates(self, x, y, extra_smooth=False):
        """Apply smoothing to coordinates."""
        self.prev_x_coords.append(x)
        self.prev_y_coords.append(y)
        
        # Check if near edges
        near_left = x < self.edge_threshold
        near_right = x > (self.screen_width - self.edge_threshold)
        near_top = y < self.edge_threshold
        near_bottom = y > (self.screen_height - self.edge_threshold)
        near_edge = near_left or near_right or near_top or near_bottom
        
        # Determine smoothing factor
        if extra_smooth or self.precision_mode_active:
            smoothing = self.precision_smoothing_factor
        elif near_edge:
            smoothing = self.edge_smoothing_factor
        else:
            smoothing = self.smoothing_factor
        
        # Apply smoothing
        if len(self.prev_x_coords) >= smoothing:
            smooth_x = int(np.mean(list(self.prev_x_coords)[-smoothing:]))
            smooth_y = int(np.mean(list(self.prev_y_coords)[-smoothing:]))
        else:
            smooth_x = int(np.mean(self.prev_x_coords))
            smooth_y = int(np.mean(self.prev_y_coords))
        
        return smooth_x, smooth_y
    
    def detect_left_hand_mode(self, fingers, landmarks=None):
        """
        Detect what mode the left hand is activating with improved accuracy.
        
        Args:
            fingers: List of finger states [thumb, index, middle, ring, pinky]
            landmarks: Optional hand landmarks for pinch detection
            
        Returns:
            ActivationMode enum value
        """
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
        fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])
        
        # Check for pinch gesture (OK sign) - SHORTCUT MODE
        if landmarks is not None:
            thumb_tip = self.get_landmark_position(landmarks, 4)
            index_tip = self.get_landmark_position(landmarks, 8)
            pinch_distance = self.calculate_distance(thumb_tip, index_tip)
            # OK gesture: thumb and index touching, other fingers can be up
            if pinch_distance < 35:
                return ActivationMode.SHORTCUT
        
        # 🖐️ All 5 fingers up = SCROLL MODE
        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return ActivationMode.SCROLL
        
        # 🤟 Thumb + Index + Pinky (rock sign) = WINDOW MODE
        # Must check this before ZOOM to avoid confusion
        if thumb_up and index_up and pinky_up and not middle_up and not ring_up:
            return ActivationMode.WINDOW
        
        # 🤘 Thumb + Pinky ONLY (Ring finger optional) = ZOOM MODE
        if thumb_up and pinky_up and not index_up and not middle_up:
            return ActivationMode.ZOOM
        
        # ✌️ Index + Middle ONLY (peace sign) = MEDIA MODE
        if index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            return ActivationMode.MEDIA
        
        # 🤙 Pinky ONLY (Ring finger optional) = VOLUME MODE
        if pinky_up and not thumb_up and not index_up and not middle_up:
            return ActivationMode.VOLUME
        
        # ✊ All fingers down (fist) = PRECISION MODE
        if not thumb_up and fingers_up_count == 0:
            return ActivationMode.PRECISION
        
        return ActivationMode.NONE
    
    def detect_right_hand_gesture(self, fingers, landmarks, left_mode):
        """
        Detect right hand gesture based on left hand mode with improved accuracy.
        
        Args:
            fingers: Right hand finger status
            landmarks: Right hand landmarks
            left_mode: Current left hand mode (ActivationMode)
            
        Returns:
            Tuple of (gesture, point1, point2)
        """
        thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
        fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])
        
        # Get fingertip positions
        thumb_tip = self.get_landmark_position(landmarks, 4)
        index_tip = self.get_landmark_position(landmarks, 8)
        middle_tip = self.get_landmark_position(landmarks, 12)
        
        # PRIORITY 1: DRAG - Very tight thumb-index pinch (Universal)
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        if thumb_index_distance < self.drag_pinch_threshold:
            return HandGesture.DRAG, index_tip, thumb_tip
        
        # MODE-SPECIFIC GESTURES
        
        # 1. SCROLL MODE - Two different gestures for up/down
        if left_mode == ActivationMode.SCROLL:
            # 1 finger (index only) = SCROLL UP
            if fingers_up_count == 1 and index_up:
                return HandGesture.ZOOM_IN, index_tip, None  # Reusing ZOOM_IN for scroll up
            # 2 fingers (index + middle) = SCROLL DOWN
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.ZOOM_OUT, index_tip, middle_tip  # Reusing ZOOM_OUT for scroll down
        
        # 2. ZOOM MODE - Discrete zoom gestures
        elif left_mode == ActivationMode.ZOOM:
            # 1 finger (index only) = ZOOM IN
            if fingers_up_count == 1 and index_up:
                return HandGesture.ZOOM_IN, index_tip, None
            # 2 fingers (index + middle) = ZOOM OUT
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.ZOOM_OUT, index_tip, middle_tip
        
        # 3. VOLUME MODE
        elif left_mode == ActivationMode.VOLUME:
            if fingers_up_count == 1 and index_up:
                return HandGesture.VOLUME_UP, index_tip, None
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.VOLUME_DOWN, index_tip, middle_tip
        
        # 4. MEDIA MODE
        elif left_mode == ActivationMode.MEDIA:
            if fingers_up_count == 1 and index_up:
                return HandGesture.MEDIA_PLAY_PAUSE, index_tip, None
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.MEDIA_NEXT, index_tip, None
            elif fingers_up_count == 3 and index_up and middle_up and ring_up:
                return HandGesture.MEDIA_PREV, index_tip, None
        
        # 5. WINDOW MODE
        elif left_mode == ActivationMode.WINDOW:
            if fingers_up_count == 1 and index_up:
                return HandGesture.MINIMIZE_WINDOW, index_tip, None
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.MAXIMIZE_WINDOW, index_tip, None
            elif fingers_up_count == 3 and index_up and middle_up and ring_up:
                return HandGesture.CLOSE_WINDOW, index_tip, None
            elif fingers_up_count == 4:
                return HandGesture.SCREENSHOT, index_tip, None
        
        # 6. SHORTCUT MODE
        elif left_mode == ActivationMode.SHORTCUT:
            if fingers_up_count == 1 and index_up:
                return HandGesture.COPY, index_tip, None
            elif fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.PASTE, index_tip, None
            elif fingers_up_count == 3 and index_up and middle_up and ring_up:
                return HandGesture.UNDO, index_tip, None
            elif fingers_up_count == 4:
                return HandGesture.SELECT_ALL, index_tip, None
        
        # STANDARD GESTURES (when no special mode active or PRECISION mode)
        if left_mode in [ActivationMode.NONE, ActivationMode.PRECISION]:
            # DOUBLE CLICK - 4 fingers up
            if fingers_up_count == 4:
                return HandGesture.DOUBLE_CLICK, index_tip, middle_tip
            
            # RIGHT CLICK - 3 fingers up
            if fingers_up_count == 3 and index_up and middle_up and ring_up:
                return HandGesture.RIGHT_CLICK, index_tip, middle_tip
            
            # LEFT CLICK - 2 fingers up (index + middle)
            if fingers_up_count == 2 and index_up and middle_up:
                return HandGesture.LEFT_CLICK, index_tip, middle_tip
            
            # MIDDLE CLICK - Thumb only
            if thumb_up and fingers_up_count == 0:
                return HandGesture.MIDDLE_CLICK, thumb_tip, None
            
            # MOVE - 1 finger up (index only)
            if fingers_up_count == 1 and index_up:
                return HandGesture.MOVE, index_tip, None
        
        return HandGesture.NONE, None, None
    
    def execute_scroll_discrete(self, direction):
        """
        Execute discrete scrolling.
        direction: 'up' or 'down'
        """
        current_time = time.time()
        
        # Add cooldown to prevent too rapid scrolling
        if hasattr(self, 'last_scroll_time'):
            if current_time - self.last_scroll_time < 0.15:  # 150ms cooldown
                return
        
        if direction == 'up':
            pyautogui.scroll(120)  # Scroll up
            print("↕ SCROLL UP ⬆")
        elif direction == 'down':
            pyautogui.scroll(-120)  # Scroll down
            print("↕ SCROLL DOWN ⬇")
        
        self.last_scroll_time = current_time
    
    def execute_zoom_discrete(self, direction):
        """
        Execute discrete zooming.
        direction: 'in' or 'out'
        """
        current_time = time.time()
        
        # Add cooldown to prevent too rapid zooming
        if hasattr(self, 'last_zoom_action_time'):
            if current_time - self.last_zoom_action_time < 0.2:  # 200ms cooldown
                return
        
        if direction == 'in':
            # Use '=' for Zoom In (Ctrl + = is standard Zoom In)
            pyautogui.hotkey('ctrl', '=')
            print("🔍 ZOOM IN")
        elif direction == 'out':
            # Use '-' for Zoom Out
            pyautogui.hotkey('ctrl', '-')
            print("🔍 ZOOM OUT")
        
        self.last_zoom_action_time = current_time
    
    def execute_volume_discrete(self, direction):
        """
        Execute discrete volume control.
        direction: 'up' or 'down'
        """
        current_time = time.time()
        
        # Add cooldown to prevent too rapid changes
        if hasattr(self, 'last_volume_time'):
            if current_time - self.last_volume_time < 0.15:  # 150ms cooldown
                return
        
        if direction == 'up':
            pyautogui.press('volumeup')
            print("🔊 VOLUME UP")
        elif direction == 'down':
            pyautogui.press('volumedown')
            print("🔉 VOLUME DOWN")
        
        self.last_volume_time = current_time
    
    def execute_click(self, gesture):
        """Execute mouse click with debouncing."""
        current_time = time.time()
        
        if current_time - self.last_click_time > self.debounce_time:
            if gesture == HandGesture.LEFT_CLICK:
                pyautogui.click()
                self.last_click_time = current_time
                print("✓ LEFT CLICK")
            elif gesture == HandGesture.RIGHT_CLICK:
                pyautogui.rightClick()
                self.last_click_time = current_time
                print("✓ RIGHT CLICK")
            elif gesture == HandGesture.MIDDLE_CLICK:
                pyautogui.middleClick()
                self.last_click_time = current_time
                print("✓ MIDDLE CLICK")
            elif gesture == HandGesture.DOUBLE_CLICK:
                pyautogui.doubleClick()
                self.last_click_time = current_time
                print("✓ DOUBLE CLICK")
    
    def execute_media(self, gesture):
        """Execute media controls."""
        current_time = time.time()
        
        if current_time - self.last_media_action_time > self.media_debounce:
            if gesture == HandGesture.MEDIA_PLAY_PAUSE:
                pyautogui.press('playpause')
                print("⏯ PLAY/PAUSE")
            elif gesture == HandGesture.MEDIA_NEXT:
                pyautogui.press('nexttrack')
                print("⏭ NEXT TRACK")
            elif gesture == HandGesture.MEDIA_PREV:
                pyautogui.press('prevtrack')
                print("⏮ PREV TRACK")
            
            self.last_media_action_time = current_time
    
    def execute_window_action(self, gesture):
        """Execute window management actions."""
        current_time = time.time()
        
        if current_time - self.last_window_action_time > self.window_debounce:
            if gesture == HandGesture.MINIMIZE_WINDOW:
                pyautogui.hotkey('win', 'down')
                print("⬇ MINIMIZE WINDOW")
            elif gesture == HandGesture.MAXIMIZE_WINDOW:
                pyautogui.hotkey('win', 'up')
                print("⬆ MAXIMIZE WINDOW")
            elif gesture == HandGesture.CLOSE_WINDOW:
                pyautogui.hotkey('alt', 'f4')
                print("✖ CLOSE WINDOW")
            elif gesture == HandGesture.SCREENSHOT:
                filename = f'screenshot_{int(time.time())}.png'
                pyautogui.screenshot(filename)
                print(f"📸 SCREENSHOT SAVED: {filename}")
            
            self.last_window_action_time = current_time
    
    def execute_shortcut(self, gesture):
        """Execute common keyboard shortcuts."""
        current_time = time.time()
        
        if current_time - self.last_shortcut_action_time > self.shortcut_debounce:
            if gesture == HandGesture.COPY:
                pyautogui.hotkey('ctrl', 'c')
                print("📋 COPY")
            elif gesture == HandGesture.PASTE:
                pyautogui.hotkey('ctrl', 'v')
                print("📄 PASTE")
            elif gesture == HandGesture.UNDO:
                pyautogui.hotkey('ctrl', 'z')
                print("↩ UNDO")
            elif gesture == HandGesture.SELECT_ALL:
                pyautogui.hotkey('ctrl', 'a')
                print("✅ SELECT ALL")
            
            self.last_shortcut_action_time = current_time
    
    def execute_drag(self, gesture, point):
        """Execute drag operation."""
        if gesture == HandGesture.DRAG:
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                self.drag_start_pos = point
                print("✓ DRAG START")
        else:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                self.drag_start_pos = None
                print("✓ DRAG END")
    
    def draw_visual_feedback(self, frame, left_mode, right_gesture, point1, point2):
        """Draw comprehensive visual feedback."""
        # Draw active region
        cv2.rectangle(
            frame,
            (self.active_x_min, self.active_y_min),
            (self.active_x_max, self.active_y_max),
            self.COLOR_ACTIVE_FRAME,
            2
        )
        cv2.putText(
            frame, "RIGHT HAND AREA", (self.active_x_min, self.active_y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_ACTIVE_FRAME, 2
        )
        
        # Draw left hand indicator
        if self.left_hand_detected:
            mode_color = self.COLOR_LEFT_HAND
            mode_text = f"LEFT: {left_mode.name}"
            
            if left_mode == ActivationMode.SCROLL:
                mode_color = self.COLOR_SCROLL_MODE
            elif left_mode == ActivationMode.ZOOM:
                mode_color = self.COLOR_ZOOM_MODE
            elif left_mode == ActivationMode.VOLUME:
                mode_color = self.COLOR_VOLUME_MODE
            elif left_mode == ActivationMode.PRECISION:
                mode_color = self.COLOR_PRECISION
            elif left_mode == ActivationMode.MEDIA:
                mode_color = (100, 255, 100)  # Light Green
            elif left_mode == ActivationMode.WINDOW:
                mode_color = (200, 200, 255)  # Light Red
            elif left_mode == ActivationMode.SHORTCUT:
                mode_color = (255, 200, 200)  # Light Blue
            
            cv2.putText(
                frame, mode_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2
            )
            cv2.putText(
                frame, "(Left Hand: Full Screen)", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
        
        # Draw right hand gesture indicator
        if self.right_hand_detected and point1:
            # Determine color and radius based on gesture
            color = self.COLOR_INDEX_TIP
            radius = 12
            text = ""
            
            if right_gesture == HandGesture.DRAG:
                color = (0, 165, 255)
                radius = 15
                text = "DRAGGING" if self.is_dragging else "DRAG READY"
            elif right_gesture == HandGesture.ZOOM_IN and left_mode == ActivationMode.SCROLL:
                color = self.COLOR_SCROLL_MODE
                radius = 15
                text = "SCROLL UP ⬆"
            elif right_gesture == HandGesture.ZOOM_OUT and left_mode == ActivationMode.SCROLL:
                color = self.COLOR_SCROLL_MODE
                radius = 15
                text = "SCROLL DOWN ⬇"
            elif right_gesture == HandGesture.SCROLL:
                color = self.COLOR_SCROLL_MODE
                radius = 15
                text = "SCROLL (Drag Style)"
            elif right_gesture == HandGesture.ZOOM_IN and left_mode == ActivationMode.ZOOM:
                color = self.COLOR_ZOOM_MODE
                radius = 15
                text = "ZOOM IN 🔍+"
            elif right_gesture == HandGesture.ZOOM_OUT and left_mode == ActivationMode.ZOOM:
                color = self.COLOR_ZOOM_MODE
                radius = 15
                text = "ZOOM OUT 🔍-"
            elif right_gesture == HandGesture.LEFT_CLICK:
                color = (0, 0, 255)
                radius = 18
                text = "LEFT CLICK"
            elif right_gesture == HandGesture.RIGHT_CLICK:
                color = (255, 0, 0)
                radius = 18
                text = "RIGHT CLICK"
            elif right_gesture == HandGesture.DOUBLE_CLICK:
                color = (0, 0, 255)
                radius = 20
                text = "DOUBLE CLICK"
            elif right_gesture == HandGesture.MIDDLE_CLICK:
                color = (0, 255, 0)
                radius = 18
                text = "MIDDLE CLICK"
            elif right_gesture == HandGesture.VOLUME_UP:
                color = self.COLOR_VOLUME_MODE
                radius = 15
                text = "VOLUME UP 🔊+"
            elif right_gesture == HandGesture.VOLUME_DOWN:
                color = self.COLOR_VOLUME_MODE
                radius = 15
                text = "VOLUME DOWN 🔉-"
            elif right_gesture in [HandGesture.MEDIA_PLAY_PAUSE, HandGesture.MEDIA_NEXT, HandGesture.MEDIA_PREV]:
                color = (100, 255, 100)
                text = right_gesture.name.replace('MEDIA_', '')
            elif right_gesture in [HandGesture.MINIMIZE_WINDOW, HandGesture.MAXIMIZE_WINDOW, 
                                   HandGesture.CLOSE_WINDOW, HandGesture.SCREENSHOT]:
                color = (200, 200, 255)
                text = right_gesture.name.replace('_', ' ')
            elif right_gesture in [HandGesture.COPY, HandGesture.PASTE, HandGesture.UNDO, HandGesture.SELECT_ALL]:
                color = (255, 200, 200)
                text = right_gesture.name
            else:
                text = "MOVE"
            
            # Draw fingertip marker
            cv2.circle(frame, point1, radius, color, cv2.FILLED)
            
            # Draw second point if exists
            if point2:
                cv2.circle(frame, point2, 12, (0, 255, 255), cv2.FILLED)
                cv2.line(frame, point1, point2, color, 3)
            
            # Draw gesture text
            cv2.putText(
                frame, text, (point1[0] - 60, point1[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        # Draw precision mode indicator
        if self.precision_mode_active:
            cv2.putText(
                frame, "PRECISION MODE", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_PRECISION, 2
            )
    
    def calculate_fps(self):
        """Calculate current FPS."""
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        return int(fps)
    
    def run(self):
        """Main execution loop."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        window_name = "AI Virtual Mouse - Two Hands v5.4"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(window_name, self.cam_width, self.cam_height)
        
        # Position window
        window_x = self.screen_width - self.cam_width - 50
        window_y = 50
        cv2.moveWindow(window_name, window_x, window_y)
        
        print("="*70)
        print("AI VIRTUAL MOUSE - TWO HANDS EDITION v5.4")
        print("="*70)
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print("")
        print("LEFT HAND MODES:")
        print("  🖐️  All 5 fingers  → SCROLL MODE")
        print("  🤘  Thumb+Pinky   → ZOOM MODE")
        print("  🤙  Pinky only    → VOLUME MODE")
        print("  ✊  Fist          → PRECISION")
        print("  ✌️  Peace sign    → MEDIA")
        print("  🤟  Rock sign     → WINDOW")
        print("  👌  OK sign       → SHORTCUTS")
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SCROLL, ZOOM & VOLUME USE THE SAME GESTURES:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  👆  1 finger (index)        = SCROLL UP / ZOOM IN / VOL UP")
        print("  ✌️  2 fingers (index+mid)   = SCROLL DOWN / ZOOM OUT / VOL DOWN")
        print("")
        print("  Left hand determines mode, right hand performs action!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("")
        print("RIGHT HAND (Standard Mode):")
        print("  👆  1 finger      → MOVE")
        print("  ✌️  2 fingers     → LEFT CLICK")
        print("  🤟  3 fingers     → RIGHT CLICK")
        print("  🖐️  4 fingers     → DOUBLE CLICK")
        print("  👍  Thumb only    → MIDDLE CLICK")
        print("  🤏  Tight pinch   → DRAG")
        print("")
        print("EXIT: Press 'q' or 'ESC'")
        print("="*70)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Failed to capture frame")
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.hands.process(rgb_frame)
                
                # Reset hand detection
                self.left_hand_detected = False
                self.right_hand_detected = False
                self.left_hand_landmarks = None
                self.right_hand_landmarks = None
                self.stored_right_fingers = None
                self.stored_right_landmarks = None
                left_mode = ActivationMode.NONE
                right_gesture = HandGesture.NONE
                point1, point2 = None, None
                
                # Process detected hands (first pass - identify hands)
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):
                        hand_label = handedness.classification[0].label
                        
                        # Draw landmarks
                        color = self.COLOR_LEFT_HAND if hand_label == 'Left' else self.COLOR_RIGHT_HAND
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Store landmarks
                        if hand_label == 'Left':
                            self.left_hand_detected = True
                            self.left_hand_landmarks = hand_landmarks.landmark
                        else:
                            self.right_hand_detected = True
                            self.right_hand_landmarks = hand_landmarks.landmark
                            # Store for later processing
                            self.stored_right_fingers = self.get_finger_status(
                                hand_landmarks.landmark, 'Right'
                            )
                            self.stored_right_landmarks = hand_landmarks.landmark
                
                # Process left hand mode first
                if self.left_hand_detected:
                    left_fingers = self.get_finger_status(self.left_hand_landmarks, 'Left')
                    left_mode = self.detect_left_hand_mode(left_fingers, self.left_hand_landmarks)
                    
                    # Update mode states
                    self.scroll_mode_active = (left_mode == ActivationMode.SCROLL)
                    self.zoom_mode_active = (left_mode == ActivationMode.ZOOM)
                    self.volume_mode_active = (left_mode == ActivationMode.VOLUME)
                    self.precision_mode_active = (left_mode == ActivationMode.PRECISION)
                    self.media_mode_active = (left_mode == ActivationMode.MEDIA)
                    self.window_mode_active = (left_mode == ActivationMode.WINDOW)
                    self.shortcut_mode_active = (left_mode == ActivationMode.SHORTCUT)
                    self.current_mode = left_mode
                
                # Process right hand gesture (now we know left_mode)
                if self.right_hand_detected and self.stored_right_landmarks:
                    right_gesture, point1, point2 = self.detect_right_hand_gesture(
                        self.stored_right_fingers,
                        self.stored_right_landmarks,
                        left_mode
                    )
                
                # Execute gestures
                if self.right_hand_detected:
                    # Handle scroll mode (discrete gestures)
                    if left_mode == ActivationMode.SCROLL:
                        # ZOOM_IN gesture = Scroll UP (1 finger)
                        if right_gesture == HandGesture.ZOOM_IN:
                            self.execute_scroll_discrete('up')
                        # ZOOM_OUT gesture = Scroll DOWN (2 fingers)
                        elif right_gesture == HandGesture.ZOOM_OUT:
                            self.execute_scroll_discrete('down')
                    
                    # Handle zoom mode (discrete gestures)
                    if left_mode == ActivationMode.ZOOM:
                        # ZOOM_IN gesture = Zoom IN (1 finger)
                        if right_gesture == HandGesture.ZOOM_IN:
                            self.execute_zoom_discrete('in')
                        # ZOOM_OUT gesture = Zoom OUT (2 fingers)
                        elif right_gesture == HandGesture.ZOOM_OUT:
                            self.execute_zoom_discrete('out')
                    
                    # Handle volume mode (discrete)
                    if left_mode == ActivationMode.VOLUME:
                        if right_gesture == HandGesture.VOLUME_UP:
                            self.execute_volume_discrete('up')
                        elif right_gesture == HandGesture.VOLUME_DOWN:
                            self.execute_volume_discrete('down')
                    
                    # Handle media mode
                    if left_mode == ActivationMode.MEDIA:
                        self.execute_media(right_gesture)
                    
                    # Handle window mode
                    if left_mode == ActivationMode.WINDOW:
                        self.execute_window_action(right_gesture)
                    
                    # Handle shortcut mode
                    if left_mode == ActivationMode.SHORTCUT:
                        self.execute_shortcut(right_gesture)
                    
                    # Handle drag
                    self.execute_drag(right_gesture, point1)
                    
                    # Handle cursor movement
                    if right_gesture in [HandGesture.MOVE, HandGesture.DRAG] and point1:
                        screen_x, screen_y = self.map_to_screen(point1[0], point1[1])
                        smooth_x, smooth_y = self.smooth_coordinates(
                            screen_x, screen_y,
                            extra_smooth=self.precision_mode_active
                        )
                        pyautogui.moveTo(smooth_x, smooth_y)
                    
                    # Handle clicks (only in standard/precision mode)
                    elif right_gesture in [HandGesture.LEFT_CLICK, HandGesture.RIGHT_CLICK, 
                                          HandGesture.MIDDLE_CLICK, HandGesture.DOUBLE_CLICK]:
                        if not self.is_dragging and left_mode in [ActivationMode.NONE, ActivationMode.PRECISION]:
                            self.execute_click(right_gesture)
                
                else:
                    # Reset states when right hand not detected
                    if self.is_dragging:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                    
                    self.last_volume_y = None
                
                # Draw visual feedback
                self.draw_visual_feedback(frame, left_mode, right_gesture, point1, point2)
                
                # Display FPS
                fps = self.calculate_fps()
                cv2.putText(
                    frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_TEXT, 2
                )
                
                # Display hand status
                hand_status = f"Hands: L:{'+' if self.left_hand_detected else '-'} R:{'+' if self.right_hand_detected else '-'}"
                cv2.putText(
                    frame, hand_status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2
                )
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Exit on 'q' or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\nExit key pressed - Shutting down...")
                    break
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt - Shutting down...")
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.is_dragging:
                pyautogui.mouseUp()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("AI Virtual Mouse Stopped")


if __name__ == "__main__":
    print("\nInitializing AI Virtual Mouse Controller...")
    print("Please ensure you have installed: opencv-python mediapipe pyautogui numpy")
    print("\nStarting in 2 seconds...\n")
    time.sleep(2)
    
    virtual_mouse = TwoHandVirtualMouse(
        cam_width=640,
        cam_height=480,
        frame_reduction=100,
        smoothing_factor=7,
        click_threshold=40,
        debounce_time=0.4
    )
    virtual_mouse.run()