import cv2
import mediapipe as mp
import numpy as np
import time

class BodyDetector:
    def __init__(self):
        # Initialize MediaPipe hands and pose
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1, # Only need one hand (the one pointing)
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        
        self.body_parts = {
            mp.solutions.pose.PoseLandmark.NOSE: "Nose",
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: "Left Shoulder",
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: "Right Shoulder",
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW: "Left Elbow",
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW: "Right Elbow",
            mp.solutions.pose.PoseLandmark.LEFT_WRIST: "Left Wrist",
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST: "Right Wrist",
            mp.solutions.pose.PoseLandmark.LEFT_HIP: "Left Hip",
            mp.solutions.pose.PoseLandmark.RIGHT_HIP: "Right Hip",
            mp.solutions.pose.PoseLandmark.LEFT_KNEE: "Left Knee",
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE: "Right Knee",
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE: "Left Ankle",
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE: "Right Ankle",
            mp.solutions.pose.PoseLandmark.LEFT_EYE: "Left Eye",
            mp.solutions.pose.PoseLandmark.RIGHT_EYE: "Right Eye",
            mp.solutions.pose.PoseLandmark.LEFT_EAR: "Left Ear",
            mp.solutions.pose.PoseLandmark.RIGHT_EAR: "Right Ear",
            mp.solutions.pose.PoseLandmark.MOUTH_LEFT: "Mouth",
            mp.solutions.pose.PoseLandmark.MOUTH_RIGHT: "Mouth",
        }
        self.detection_threshold = 0.05 # Adjust based on how close finger needs to be (normalized coords)
        
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process for pose and hands
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            detected_part = "None"
            index_finger_tip = None

            # Draw pose landmarks and get body landmark positions
            body_landmarks_positions = {}
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                h, w, c = frame.shape
                for landmark_enum, landmark_name in self.body_parts.items():
                    landmark = pose_results.pose_landmarks.landmark[landmark_enum]
                    body_landmarks_positions[landmark_name] = (int(landmark.x * w), int(landmark.y * h))

            # Draw hand landmarks and get index finger tip position
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    h, w, c = frame.shape
                    index_finger_tip = (int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),\
                                        int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
                    # Assuming only one hand is used for pointing, we can break after finding one
                    break

            # Check for proximity of index finger tip to body landmarks
            if index_finger_tip and body_landmarks_positions:
                index_tip_x, index_tip_y = index_finger_tip
                
                for part_name, (lx, ly) in body_landmarks_positions.items():
                    # Calculate distance in pixels
                    distance = np.sqrt((index_tip_x - lx)**2 + (index_tip_y - ly)**2)
                    
                    # Check if distance is within the threshold (adjust threshold as needed)
                    # A fixed pixel threshold might be better here or a dynamic one based on body size in frame.
                    # Let's use a fixed pixel threshold for simplicity initially, say 50 pixels.
                    if distance < 50: # You might need to adjust this pixel threshold
                         detected_part = part_name
                         break # Found the closest body part, stop checking others

            # Display the detected body part name
            cv2.putText(frame, f'Touching: {detected_part}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Display the frame
            cv2.imshow('Body Part Detector', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = BodyDetector()
    detector.run() 