import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import time

@dataclass
class HandGesture:
    name: str
    color: Tuple[int, int, int]
    confidence: float

class HandTracker:
    def __init__(self,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.palm_landmarks = [0, 1, 5, 9, 13, 17]
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]
        
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def draw_labeled_box(self, image: np.ndarray, text: str, position: Tuple[int, int], 
                        color: Tuple[int, int, int]) -> None:
        """Draw a labeled box with gesture information"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate box coordinates
        x, y = position
        box_coords = [
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding)
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw border
        cv2.rectangle(image, box_coords[0], box_coords[1], color, thickness)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    def get_landmark_coordinates(self, landmarks, idx) -> np.ndarray:
        return np.array([landmarks.landmark[idx].x, landmarks.landmark[idx].y, landmarks.landmark[idx].z])

    def calculate_finger_angles(self, landmarks) -> Dict[str, float]:
        angles = {}
        
        finger_landmarks = {
            'Thumb': [2, 3, 4],
            'Index': [5, 6, 8],
            'Middle': [9, 10, 12],
            'Ring': [13, 14, 16],
            'Pinky': [17, 18, 20]
        }
        
        for finger, points in finger_landmarks.items():
            p1 = self.get_landmark_coordinates(landmarks, points[0])
            p2 = self.get_landmark_coordinates(landmarks, points[1])
            p3 = self.get_landmark_coordinates(landmarks, points[2])
            
            v1 = p1 - p2
            v2 = p3 - p2
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
            angles[finger] = angle
            
        return angles

    def calculate_finger_distances(self, landmarks) -> Dict[str, float]:
        palm_center = np.mean([self.get_landmark_coordinates(landmarks, idx) 
                             for idx in self.palm_landmarks], axis=0)
        
        distances = {}
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for name, tip_idx in zip(finger_names, self.finger_tips):
            tip_pos = self.get_landmark_coordinates(landmarks, tip_idx)
            distance = np.linalg.norm(tip_pos - palm_center)
            distances[name] = distance
            
        return distances

    def is_finger_closed(self, landmarks, finger_tip: int, finger_pip: int, 
                        palm_ref: int, threshold: float = 0.1) -> bool:
        tip = self.get_landmark_coordinates(landmarks, finger_tip)
        pip = self.get_landmark_coordinates(landmarks, finger_pip)
        palm = self.get_landmark_coordinates(landmarks, palm_ref)
        
        return np.linalg.norm(tip - palm) < np.linalg.norm(pip - palm) + threshold

    def is_thumb_pointing_down(self, landmarks) -> bool:
        """Check if thumb is pointing downward"""
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        wrist = landmarks.landmark[0]
        
        # Check if thumb tip is below the MCP joint and pointing downward
        return (thumb_tip.y > thumb_mcp.y and 
                thumb_tip.y > wrist.y and 
                abs(thumb_tip.x - thumb_mcp.x) < 0.1)  # Ensure thumb is relatively straight

    def detect_gesture(self, landmarks, angles: Dict[str, float]) -> Optional[HandGesture]:
        distances = self.calculate_finger_distances(landmarks)
        
        # Check for thumbs down
        is_thumbs_down = (self.is_thumb_pointing_down(landmarks) and 
                         all(self.is_finger_closed(landmarks, tip, pip, 0)
                             for tip, pip in zip(self.finger_tips[1:], self.finger_pips[1:])))
        if is_thumbs_down:
            return HandGesture("Thumbs Down", (255, 0, 255), 0.9)  # Purple color
        
        # Check for open hand
        is_open = all(angle > 150 for angle in angles.values()) and \
                  all(dist > 0.15 for dist in distances.values())
        if is_open:
            return HandGesture("Open Hand", (0, 255, 0), 0.9)  # Green color
        
        # Check for peace sign
        is_peace = (angles['Index'] > 150 and angles['Middle'] > 150 and 
                   all(angles[finger] < 90 for finger in ['Ring', 'Pinky']) and
                   distances['Index'] > 0.15 and distances['Middle'] > 0.15)
        if is_peace:
            return HandGesture("Peace Sign", (255, 255, 0), 0.85)  # Yellow color
        
        # Check for thumbs up
        is_thumbs_up = (angles['Thumb'] > 150 and 
                       all(self.is_finger_closed(landmarks, tip, pip, 0)
                           for tip, pip in zip(self.finger_tips[1:], self.finger_pips[1:])))
        if is_thumbs_up:
            return HandGesture("Thumbs Up", (255, 0, 0), 0.9)  # Red color
        
        return None

    def draw_fps(self, image: np.ndarray) -> None:
        """Draw FPS counter in a box"""
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        
        fps_text = f"FPS: {int(fps)}"
        h, _, _ = image.shape
        self.draw_labeled_box(image, fps_text, (10, h - 30), (0, 255, 0))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, 
                                                               results.multi_handedness)):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                angles = self.calculate_finger_angles(hand_landmarks)
                gesture = self.detect_gesture(hand_landmarks, angles)
                
                if gesture:
                    hand_label = "Right" if handedness.classification[0].label == "Left" else "Left"
                    text = f"{hand_label} Hand: {gesture.name} ({gesture.confidence:.2f})"
                    # Position boxes for multiple hands
                    y_position = 40 + i * 60
                    self.draw_labeled_box(image, text, (10, y_position), gesture.color)
        
        # Draw FPS counter
        self.draw_fps(image)
        
        return image

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = tracker.process_frame(frame)
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()