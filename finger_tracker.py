import cv2
import mediapipe as mp
import numpy as np
import time

class FingerTracker:
    def __init__(self):
        # Initialize MediaPipe hands (allow up to 2 hands)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        
        # Drawing canvas
        # Get frame dimensions first
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture video stream.")
            exit()
            
        self.frame_height, self.frame_width, _ = frame.shape
        self.drawing_canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        self.drawing_mode = False
        self.last_point = None
        self.fist_toggle_timer = None # Timer for intentional fist toggle
        self.fist_toggle_duration = 1.0 # Duration in seconds to hold fist for toggle
        
    def count_fingers(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        
        fingers_up = 0
        # Thumb
        if landmarks[4][0] < landmarks[3][0]: # This is more aligned with right hand in mirror view (left side of frame)
             # Need a better handedness-aware thumb check for true accuracy.
             # For now, let's rely on relative landmark positions for other fingers
             pass # Rely more on relative tip-to-base for other fingers and overall landmarks
             
        # Count other fingers by checking if the tip is above the second joint (y-coordinate is smaller)
        # These checks are more reliable than the simple thumb x-check without handedness logic.
        for tip_id in [8, 12, 16, 20]: # Index, Middle, Ring, Pinky
            if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
                fingers_up += 1

        # Revisit thumb counting - a more accurate way considering handedness and hand orientation
        # Check if thumb tip is significantly further from palm base than other collapsed fingers
        # This is complex. Let's stick to a simplified robust check:
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        index_mcp = landmarks[5] # Metacarpophalangeal joint of index finger

        # A simple check: is the thumb tip significantly to the left or right of the thumb base (considering handedness)?
        # Or is it extended away from the palm center?
        # Given the mirror view, 'Right' hand in results is user's left hand, 'Left' hand is user's right.
        # For user's right hand ('Left' in results), thumb extends to the left in mirror view.
        # For user's left hand ('Right' in results), thumb extends to the right in mirror view.

        # Simpler check: is thumb tip's y-coordinate above the index finger's MCP joint's y-coordinate?
        # This works for many upright hand poses.
        if thumb_tip[1] < index_mcp[1]:
             # Add a check to ensure the thumb is actually extended horizontally away from the palm area
             # This is still an approximation without full 3D orientation
             # A simple distance check from thumb tip to palm center might work better
             palm_center = landmarks[0] # Wrist landmark as a proxy for palm center
             dist_thumb_to_center = np.sqrt((thumb_tip[0] - palm_center[0])**2 + (thumb_tip[1] - palm_center[1])**2)
             dist_index_mcp_to_center = np.sqrt((index_mcp[0] - palm_center[0])**2 + (index_mcp[1] - palm_center[1])**2)
             
             # If thumb is further from the palm center than the index MCP, it's likely up
             if dist_thumb_to_center > dist_index_mcp_to_center * 1.2: # Use a factor to avoid false positives
                 fingers_up += 1
                
        return fingers_up

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Flip the frame horizontally for a selfie-view.
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands.
            results = self.hands.process(rgb_frame)
            
            left_hand_index_tip = None # To store the left index finger tip coordinates
            right_hand_fist = False # To check if right hand is a fist
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks for all detected hands
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Check handedness
                    if handedness.classification[0].label == 'Left': # User's right hand
                        fingers = self.count_fingers(hand_landmarks)
                        if fingers == 0:
                            right_hand_fist = True

                    elif handedness.classification[0].label == 'Right': # User's left hand
                         # Get left index finger tip coordinates
                         h, w, c = frame.shape
                         left_hand_index_tip = (int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),\
                                                 int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))

            # Toggle drawing mode with a right fist
            now = time.time()
            if right_hand_fist:
                if self.fist_toggle_timer is None:
                    self.fist_toggle_timer = now
                elif now - self.fist_toggle_timer > self.fist_toggle_duration: # Hold fist for specified duration to toggle
                    self.drawing_mode = not self.drawing_mode
                    self.last_point = None # Reset last point when toggling mode
                    self.fist_toggle_timer = None # Reset timer
            else:
                self.fist_toggle_timer = None # Reset timer if fist is released

            # Drawing logic using the left index finger tip
            if self.drawing_mode and left_hand_index_tip is not None:
                 if self.last_point is not None:
                      # Draw a line from the last point to the current point
                      cv2.line(self.drawing_canvas, self.last_point, left_hand_index_tip, (0, 255, 0), 5)
                 self.last_point = left_hand_index_tip
            else:
                 # Stop drawing if not in drawing mode or no left index finger detected
                 self.last_point = None

            # Combine the camera feed and the drawing canvas
            combined_frame = cv2.addWeighted(frame, 1, self.drawing_canvas, 0.5, 0)

            # Display drawing mode status
            mode_text = "Drawing ON" if self.drawing_mode else "Drawing OFF"
            cv2.putText(combined_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the combined frame
            cv2.imshow('Finger Tracker', combined_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear the drawing canvas
                self.drawing_canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FingerTracker()
    tracker.run() 