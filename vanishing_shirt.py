import cv2
import mediapipe as mp
import numpy as np
import time

class VanishingShirt:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Allow detecting both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        
        # Background capture setup
        self.background_frame = None
        self.background_capture_time = None
        self.capture_countdown_start = None
        self.initial_capture_delay = 3 # seconds
        self.auto_vanish_delay = 5 # seconds after background capture to turn on vanish mode

        self.vanish_mode = False
        self.palm_toggle_timer = None # Timer for intentional palm toggle
        self.palm_toggle_duration = 0.7 # Duration to hold palm to toggle vanish mode

    def capture_background(self):
        print(f"Capturing background in {self.initial_capture_delay} seconds. Please step out of the frame.")
        self.capture_countdown_start = time.time()
        # Wait for the countdown to finish in the run loop
        self.background_frame = None # Clear previous background
        self.background_capture_time = None

    def process_background_capture(self, frame):
        now = time.time()
        elapsed = now - self.capture_countdown_start
        remaining = max(0, int(self.initial_capture_delay - elapsed))
        
        # Display countdown
        cv2.putText(frame, f'Capturing in: {remaining}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if elapsed >= self.initial_capture_delay:
            ret, captured_frame = self.cap.read()
            if ret:
                self.background_frame = cv2.flip(captured_frame, 1)
                self.background_capture_time = time.time() # Record time of capture
                self.capture_countdown_start = None # Stop countdown display
                print("Background captured.")
            else:
                print("Failed to capture background frame.")
                # Potentially exit or show an error message
                
        return frame # Return frame with countdown drawn


    def are_all_fingers_down(self, landmarks):
        # Check if all finger tips are below their respective base joints (y-coordinate)
        # This is a common way to detect a closed fist or palm facing forward/backward.
        # Thumb tip landmark 4, base landmark 2
        # Index tip 8, base 5
        # Middle tip 12, base 9
        # Ring tip 16, base 13
        # Pinky tip 20, base 17
        
        if len(landmarks) < 21:
            return False

        # Check thumb separately (can be tricky based on orientation)
        # A simple check: is thumb tip's y-coordinate below thumb base y-coordinate
        thumb_down = landmarks[4].y > landmarks[2].y

        # Check other fingers: tip y below base y
        index_down = landmarks[8].y > landmarks[5].y
        middle_down = landmarks[12].y > landmarks[9].y
        ring_down = landmarks[16].y > landmarks[13].y
        pinky_down = landmarks[20].y > landmarks[17].y

        return thumb_down and index_down and middle_down and ring_down and pinky_down

    def run(self):
        self.capture_background() # Start the initial background capture process

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # If capturing background, display countdown and capture when ready
            if self.capture_countdown_start is not None:
                display_frame = self.process_background_capture(frame.copy()) # Process and get frame with countdown

            elif self.background_frame is not None: # Background is captured, proceed with main logic

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                palm_detected_in_frame = False
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Check for palm gesture on this hand
                        if self.are_all_fingers_down(hand_landmarks.landmark):
                            palm_detected_in_frame = True
                            # No need to check other hands if one palm is found
                            break

                # Auto turn on vanish mode after delay
                now = time.time()
                if not self.vanish_mode and self.background_capture_time is not None and now - self.background_capture_time >= self.auto_vanish_delay:
                     self.vanish_mode = True
                     print("Auto van ish mode ON.")

                # Toggle vanish mode with a palm (only after auto turn on delay or if already on)
                if now - self.background_capture_time >= self.auto_vanish_delay or self.vanish_mode:
                    if palm_detected_in_frame:
                        if self.palm_toggle_timer is None:
                            self.palm_toggle_timer = now
                        elif now - self.palm_toggle_timer > self.palm_toggle_duration: # Hold palm for duration to toggle
                            self.vanish_mode = not self.vanish_mode
                            self.palm_toggle_timer = None # Reset timer
                            print(f"Vanish mode toggled: {self.vanish_mode}")
                    else:
                        self.palm_toggle_timer = None # Reset timer if palm is released

                # Apply vanishing effect if enabled
                display_frame = frame.copy() # Start with the current frame
                if self.vanish_mode:
                    # Basic background subtraction (difference)
                    diff = cv2.absdiff(frame, self.background_frame)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    # Adjust threshold based on lighting and shirt/background difference
                    _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
                    
                    # Invert the mask
                    mask_inv = cv2.bitwise_not(mask)

                    # Black-out the area of person/shirt in the current frame
                    frame_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)
                    
                    # Take only background region from background image
                    background_fg = cv2.bitwise_and(self.background_frame, self.background_frame, mask = mask)
                    
                    # Combine the two images
                    display_frame = cv2.add(frame_bg, background_fg)
                    
                    # Display status on the frame
                    cv2.putText(display_frame, "VANISH ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                     # Display status on the frame
                    cv2.putText(display_frame, "VANISH OFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else: # If background is not captured yet and not in countdown (shouldn't happen with initial call)
                display_frame = frame # Just show the live feed
                cv2.putText(display_frame, "Waiting for background capture...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


            # Display the frame
            cv2.imshow('Vanish Shirt', display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                # Recapture background
                self.capture_background()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vanisher = VanishingShirt()
    vanisher.run() 