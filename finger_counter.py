import cv2
import mediapipe as mp
import numpy as np
import time

class FingerCounter:
    def __init__(self):
        # Initialize MediaPipe hands
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
        
        # Finger tip IDs
        self.finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self.thumb_tip = 4
        
        # State machine
        self.state = 'WAIT_FIRST'  # WAIT_FIRST, WAIT_OP, WAIT_SECOND, SHOW_RESULT
        self.first_number = None
        self.second_number = None
        self.operation = None
        self.result = None
        self.steady_start = None
        self.last_finger_count = None
        self.message = 'Show first number (hold fingers steady for 3s)'
        self.steady_time_required = 3  # seconds
        self.last_op_detected = None
        self.op_steady_start = None
        self.fist_start = None  # For fist reset
        self.fist_time_required = 1  # seconds
        self.result_time = None  # For lockout after result
        self.result_lockout = 2  # seconds
        
    def count_fingers(self, hand_landmarks):
        # Get all landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        
        # Count raised fingers
        fingers_up = 0
        
        # Check thumb (special case)
        if landmarks[self.thumb_tip][0] < landmarks[self.thumb_tip - 1][0]:
            fingers_up += 1
            
        # Check other fingers
        for tip_id in self.finger_tips[1:]:  # Skip thumb
            if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
                fingers_up += 1
                
        return fingers_up, landmarks

    def thumb_down(self, landmarks):
        # Thumb tip below index base (y), or close to palm (x)
        return landmarks[4][1] > landmarks[5][1]

    def detect_plus_sign(self, landmarks):
        # Crossed index and middle fingers, thumb down
        if len(landmarks) < 21:
            return False
        if not self.thumb_down(landmarks):
            return False
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_up = landmarks[16][1] < landmarks[14][1]
        pinky_up = landmarks[20][1] < landmarks[18][1]
        if index_up and middle_up and not ring_up and not pinky_up:
            ix, iy = landmarks[8]
            mx, my = landmarks[12]
            dist = np.sqrt((ix - mx) ** 2 + (iy - my) ** 2)
            if dist < 0.03:
                return True
        return False

    def detect_minus_sign(self, landmarks):
        # Only index finger up, thumb down
        if len(landmarks) < 21:
            return False
        if not self.thumb_down(landmarks):
            return False
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_up = landmarks[16][1] < landmarks[14][1]
        pinky_up = landmarks[20][1] < landmarks[18][1]
        if index_up and not middle_up and not ring_up and not pinky_up:
            return True
        return False

    def detect_multiply_sign(self, landmarks):
        # V sign: index and middle up, spread apart, not crossed, thumb down
        if len(landmarks) < 21:
            return False
        if not self.thumb_down(landmarks):
            return False
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_up = landmarks[16][1] < landmarks[14][1]
        pinky_up = landmarks[20][1] < landmarks[18][1]
        if index_up and middle_up and not ring_up and not pinky_up:
            ix, iy = landmarks[8]
            mx, my = landmarks[12]
            dist = np.sqrt((ix - mx) ** 2 + (iy - my) ** 2)
            if dist > 0.09:
                return True
        return False

    def detect_divide_sign(self, landmarks):
        # Only pinky finger up, thumb down
        if len(landmarks) < 21:
            return False
        if not self.thumb_down(landmarks):
            return False
        pinky_up = landmarks[20][1] < landmarks[18][1]
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_up = landmarks[16][1] < landmarks[14][1]
        if pinky_up and not index_up and not middle_up and not ring_up:
            return True
        return False

    def draw_count(self, frame, count, message):
        cv2.rectangle(frame, (20, 20), (700, 160), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f'Fingers: {count}',
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            4
        )
        cv2.putText(
            frame,
            message,
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            total_fingers = 0
            all_landmarks = []
            hands_detected = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    count, landmarks = self.count_fingers(hand_landmarks)
                    total_fingers += count
                    all_landmarks.append(landmarks)
                    hands_detected += 1
            now = time.time()
            if self.state == 'WAIT_FIRST':
                self.fist_start = None
                self.result_time = None
                if hands_detected > 0:
                    if self.last_finger_count == total_fingers:
                        if self.steady_start is None:
                            self.steady_start = now
                        elif now - self.steady_start >= self.steady_time_required:
                            self.first_number = total_fingers
                            self.state = 'WAIT_OP'
                            self.message = f'First number recorded: {self.first_number}. Show operation gesture.'
                            self.steady_start = None
                            self.last_finger_count = None
                    else:
                        self.steady_start = now
                        self.last_finger_count = total_fingers
                else:
                    self.steady_start = None
                    self.last_finger_count = None
            elif self.state == 'WAIT_OP':
                self.fist_start = None
                op = None
                for i, landmarks in enumerate(all_landmarks):
                    if self.detect_plus_sign(landmarks):
                        op = '+'
                        break
                    if self.detect_minus_sign(landmarks):
                        op = '-'
                        break
                    if self.detect_multiply_sign(landmarks):
                        op = '*'
                        break
                    if self.detect_divide_sign(landmarks):
                        op = '/'
                        break
                if op is None:
                    self.op_steady_start = None
                    self.last_op_detected = None
                else:
                    if self.last_op_detected == op:
                        if now - self.op_steady_start >= 1.5:
                            self.operation = op
                            self.state = 'WAIT_SECOND'
                            op_name = {'+': 'plus', '-': 'minus', '*': 'multiply', '/': 'divide'}[op]
                            self.message = f'Operation: {op} ({op_name}). Show second number (hold steady for 3s)'
                            self.op_steady_start = None
                            self.last_op_detected = None
                    else:
                        self.op_steady_start = now
                        self.last_op_detected = op
            elif self.state == 'WAIT_SECOND':
                self.fist_start = None
                if hands_detected > 0:
                    if self.last_finger_count == total_fingers:
                        if self.steady_start is None:
                            self.steady_start = now
                        elif now - self.steady_start >= self.steady_time_required:
                            self.second_number = total_fingers
                            self.state = 'SHOW_RESULT'
                            if self.operation == '+':
                                self.result = self.first_number + self.second_number
                            elif self.operation == '-':
                                self.result = self.first_number - self.second_number
                            elif self.operation == '*':
                                self.result = self.first_number * self.second_number
                            elif self.operation == '/':
                                if self.second_number == 0:
                                    self.result = 'undefined'
                                else:
                                    self.result = round(self.first_number / self.second_number, 2)
                            self.message = f'Second number: {self.second_number}. Result: {self.first_number} {self.operation} {self.second_number} = {self.result}'
                            self.steady_start = None
                            self.last_finger_count = None
                            self.result_time = now
                    else:
                        self.steady_start = now
                        self.last_finger_count = total_fingers
                else:
                    self.steady_start = None
                    self.last_finger_count = None
            elif self.state == 'SHOW_RESULT':
                # Only reset if a fist (0 fingers) is shown for 1s AND at least one hand is visible, and at least 2s have passed since result
                if self.result_time is not None and now - self.result_time < self.result_lockout:
                    pass  # lockout, do nothing
                elif hands_detected > 0 and total_fingers == 0:
                    if self.fist_start is None:
                        self.fist_start = now
                    elif now - self.fist_start >= self.fist_time_required:
                        self.state = 'WAIT_FIRST'
                        self.first_number = None
                        self.second_number = None
                        self.operation = None
                        self.result = None
                        self.message = 'Show first number (hold fingers steady for 3s)'
                        self.fist_start = None
                        self.result_time = None
                else:
                    self.fist_start = None
            self.draw_count(frame, total_fingers, self.message)
            cv2.imshow('Finger Counter', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = FingerCounter()
    counter.run() 