"""
AI Virtual Mouse Controller
A real-time hand gesture-based mouse controller using computer vision.

Author: Senior Python Developer & CV Engineer
Date: December 2024
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


class VirtualMouse:
    """
    AI-powered virtual mouse controller using hand gestures.
    
    Features:
    - Real-time hand tracking with MediaPipe
    - Smooth cursor movement with configurable smoothing
    - Gesture-based clicking with debouncing
    - Visual feedback and performance monitoring
    """
    
    def __init__(self, 
                 cam_width=640, 
                 cam_height=480,
                 frame_reduction=100,
                 smoothing_factor=7,
                 click_threshold=30,
                 debounce_time=0.3,
                 dominant_hand='Right'):
        """
        Initialize the Virtual Mouse system.
        
        Args:
            cam_width: Camera capture width
            cam_height: Camera capture height
            frame_reduction: Pixels to reduce from frame edges (creates active region)
            smoothing_factor: Number of previous coordinates to average for smoothing
            click_threshold: Distance threshold (pixels) for click detection
            debounce_time: Minimum time (seconds) between clicks
            dominant_hand: 'Right' or 'Left' hand to track
        """
        # Camera settings
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Active region settings (frame within camera view that maps to full screen)
        self.frame_reduction = frame_reduction
        self.active_x_min = frame_reduction
        self.active_x_max = cam_width - frame_reduction
        self.active_y_min = frame_reduction
        self.active_y_max = cam_height - frame_reduction
        
        # Smoothing buffer for cursor movement
        self.smoothing_factor = smoothing_factor
        self.prev_x_coords = deque(maxlen=smoothing_factor)
        self.prev_y_coords = deque(maxlen=smoothing_factor)
        
        # Click detection settings
        self.click_threshold = click_threshold
        self.debounce_time = debounce_time
        self.last_click_time = 0
        self.last_gesture = HandGesture.NONE
        
        # Hand tracking settings
        self.dominant_hand = dominant_hand
        
        # MediaPipe Hand tracking initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # FPS calculation
        self.prev_time = 0
        self.curr_time = 0
        
        # PyAutoGUI settings for faster response
        pyautogui.FAILSAFE = False  # Disable failsafe for smoother operation
        pyautogui.PAUSE = 0  # Remove delay between PyAutoGUI commands
        
        # Visual feedback colors (BGR format)
        self.COLOR_ACTIVE_FRAME = (0, 255, 0)  # Green
        self.COLOR_INDEX_TIP = (255, 0, 255)  # Magenta
        self.COLOR_MIDDLE_TIP = (0, 255, 255)  # Yellow
        self.COLOR_CLICK_LINE = (0, 0, 255)  # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        
    def get_finger_status(self, landmarks):
        """
        Determine which fingers are up or down.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            List of 5 boolean values [Thumb, Index, Middle, Ring, Pinky]
            True = finger is up, False = finger is down
        """
        fingers = []
        
        # Landmark indices for fingertips and joints
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        
        # Thumb: Special case (compare x-coordinate for right/left hand)
        if self.dominant_hand == 'Right':
            # For right hand: thumb is up if tip is to the right of IP joint
            fingers.append(landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x)
        else:
            # For left hand: thumb is up if tip is to the left of IP joint
            fingers.append(landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x)
        
        # Other fingers: Compare y-coordinate (tip vs PIP joint)
        # Finger is up if tip y-coordinate is less than PIP joint y-coordinate
        for id in range(1, 5):
            fingers.append(landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y)
            
        return fingers
    
    def get_landmark_position(self, landmarks, landmark_id):
        """
        Get pixel coordinates of a specific landmark.
        
        Args:
            landmarks: MediaPipe hand landmarks
            landmark_id: ID of the landmark (0-20)
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        landmark = landmarks[landmark_id]
        x = int(landmark.x * self.cam_width)
        y = int(landmark.y * self.cam_height)
        return x, y
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: Tuple of (x1, y1)
            point2: Tuple of (x2, y2)
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def map_to_screen(self, cam_x, cam_y):
        """
        Map camera coordinates to screen coordinates with interpolation.
        
        This creates a proportional mapping where:
        - Active region edges map to screen edges
        - Coordinates are interpolated linearly
        - Values are clamped to screen boundaries
        
        Args:
            cam_x: X coordinate in camera frame
            cam_y: Y coordinate in camera frame
            
        Returns:
            Tuple of (screen_x, screen_y)
        """
        # Linear interpolation formula:
        # screen_coord = (cam_coord - active_min) / (active_max - active_min) * screen_size
        
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
        
        # Clamp values to screen boundaries
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return int(screen_x), int(screen_y)
    
    def smooth_coordinates(self, x, y):
        """
        Apply smoothing to coordinates to reduce jitter.
        
        Uses a moving average filter over the last N coordinates.
        
        Args:
            x: Current x coordinate
            y: Current y coordinate
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        self.prev_x_coords.append(x)
        self.prev_y_coords.append(y)
        
        # Calculate average of stored coordinates
        smooth_x = int(np.mean(self.prev_x_coords))
        smooth_y = int(np.mean(self.prev_y_coords))
        
        return smooth_x, smooth_y
    
    def detect_gesture(self, fingers, landmarks):
        """
        Detect the current hand gesture based on finger positions.
        
        Gesture Logic:
        - MOVE: Index finger up, Middle finger down
        - LEFT_CLICK: Index and Middle fingers up AND close together
        - RIGHT_CLICK: Index, Middle, and Ring fingers up
        
        Args:
            fingers: List of finger up/down status
            landmarks: MediaPipe hand landmarks
            
        Returns:
            HandGesture enum value and relevant coordinates
        """
        index_up = fingers[1]
        middle_up = fingers[2]
        ring_up = fingers[3]
        
        # Get fingertip positions
        index_tip = self.get_landmark_position(landmarks, 8)  # Index fingertip
        middle_tip = self.get_landmark_position(landmarks, 12)  # Middle fingertip
        
        # Movement State: Only index finger up
        if index_up and not middle_up:
            return HandGesture.MOVE, index_tip, None
        
        # Potential Click State: Index and Middle fingers up
        elif index_up and middle_up:
            distance = self.calculate_distance(index_tip, middle_tip)
            
            # Left Click: Fingers are close together (pinch gesture)
            if distance < self.click_threshold:
                return HandGesture.LEFT_CLICK, index_tip, middle_tip
            
            # Right Click: Index, Middle, AND Ring fingers up
            # Note: This is a more ergonomic alternative to thumb pinch
            # Three fingers up is easier to detect and perform consistently
            elif ring_up:
                return HandGesture.RIGHT_CLICK, index_tip, middle_tip
            
            # Fingers up but not close enough for click
            else:
                return HandGesture.MOVE, index_tip, middle_tip
        
        return HandGesture.NONE, None, None
    
    def execute_click(self, gesture):
        """
        Execute mouse click with debouncing to prevent multiple clicks.
        
        Args:
            gesture: HandGesture enum value
        """
        current_time = time.time()
        
        # Check if enough time has passed since last click (debouncing)
        if current_time - self.last_click_time > self.debounce_time:
            if gesture == HandGesture.LEFT_CLICK:
                pyautogui.click()
                self.last_click_time = current_time
                print("LEFT CLICK")
            elif gesture == HandGesture.RIGHT_CLICK:
                pyautogui.rightClick()
                self.last_click_time = current_time
                print("RIGHT CLICK")
    
    def draw_visual_feedback(self, frame, gesture, point1, point2):
        """
        Draw visual feedback on the frame.
        
        Args:
            frame: Video frame to draw on
            gesture: Current HandGesture
            point1: Primary point (usually index fingertip)
            point2: Secondary point (usually middle fingertip)
        """
        # Draw active region rectangle
        cv2.rectangle(
            frame, 
            (self.active_x_min, self.active_y_min), 
            (self.active_x_max, self.active_y_max),
            self.COLOR_ACTIVE_FRAME, 
            2
        )
        
        # Draw fingertip markers
        if point1 is not None:
            cv2.circle(frame, point1, 10, self.COLOR_INDEX_TIP, cv2.FILLED)
        
        if point2 is not None:
            cv2.circle(frame, point2, 10, self.COLOR_MIDDLE_TIP, cv2.FILLED)
        
        # Draw line between fingers for click gestures
        if gesture in [HandGesture.LEFT_CLICK, HandGesture.RIGHT_CLICK] and point1 and point2:
            cv2.line(frame, point1, point2, self.COLOR_CLICK_LINE, 3)
            
            # Draw click indicator text
            mid_x = (point1[0] + point2[0]) // 2
            mid_y = (point1[1] + point2[1]) // 2
            
            click_text = "LEFT CLICK" if gesture == HandGesture.LEFT_CLICK else "RIGHT CLICK"
            cv2.putText(
                frame, 
                click_text, 
                (mid_x - 60, mid_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                self.COLOR_CLICK_LINE, 
                2
            )
    
    def calculate_fps(self):
        """
        Calculate current frames per second.
        
        Returns:
            FPS value as integer
        """
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        return int(fps)
    
    def run(self):
        """
        Main execution loop for the virtual mouse system.
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("AI Virtual Mouse Started")
        print(f"Screen Resolution: {self.screen_width}x{self.screen_height}")
        print(f"Tracking: {self.dominant_hand} hand")
        print("Gestures:")
        print("  - Index finger up: Move cursor")
        print("  - Index + Middle close together: Left click")
        print("  - Index + Middle + Ring up: Right click")
        print("Press 'q' to quit")
        
        try:
            while True:
                success, frame = cap.read()
                
                if not success:
                    print("Error: Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.hands.process(rgb_frame)
                
                gesture = HandGesture.NONE
                point1, point2 = None, None
                
                # Check if hand is detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Get finger status
                        fingers = self.get_finger_status(hand_landmarks.landmark)
                        
                        # Detect gesture
                        gesture, point1, point2 = self.detect_gesture(
                            fingers, 
                            hand_landmarks.landmark
                        )
                        
                        # Handle cursor movement
                        if gesture == HandGesture.MOVE and point1:
                            # Map camera coordinates to screen coordinates
                            screen_x, screen_y = self.map_to_screen(point1[0], point1[1])
                            
                            # Apply smoothing
                            smooth_x, smooth_y = self.smooth_coordinates(screen_x, screen_y)
                            
                            # Move mouse cursor
                            pyautogui.moveTo(smooth_x, smooth_y)
                        
                        # Handle clicks
                        elif gesture in [HandGesture.LEFT_CLICK, HandGesture.RIGHT_CLICK]:
                            self.execute_click(gesture)
                
                # Draw visual feedback
                self.draw_visual_feedback(frame, gesture, point1, point2)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                cv2.putText(
                    frame, 
                    f"FPS: {fps}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    self.COLOR_TEXT, 
                    2
                )
                
                # Display gesture mode
                mode_text = f"Mode: {gesture.name}"
                cv2.putText(
                    frame, 
                    mode_text, 
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    self.COLOR_TEXT, 
                    2
                )
                
                # Show frame
                cv2.imshow("AI Virtual Mouse", frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error occurred: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("AI Virtual Mouse Stopped")


if __name__ == "__main__":
    # Create and run virtual mouse with optimized settings
    virtual_mouse = VirtualMouse(
        cam_width=640,
        cam_height=480,
        frame_reduction=100,      # Active region padding
        smoothing_factor=5,       # Balance between smoothness and responsiveness
        click_threshold=30,       # Distance for click detection
        debounce_time=0.3,        # Prevent double-clicks
        dominant_hand='Right'     # Change to 'Left' if needed
    )
    
    virtual_mouse.run()
