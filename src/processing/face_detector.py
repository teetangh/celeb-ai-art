"""Face detection and landmark extraction utilities."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import face_recognition
from pathlib import Path

from ..core.models import FaceDetection


class FaceDetector:
    """Handles face detection and landmark extraction."""
    
    def __init__(self):
        """Initialize the face detector."""
        # Load OpenCV's face cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Parameters for face detection
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
    
    def detect_faces_opencv(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV (faster but less accurate).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            return [tuple(face) for face in faces]
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def detect_faces_dlib(self, image_path: str) -> List[FaceDetection]:
        """
        Detect faces using face_recognition library (more accurate).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of FaceDetection objects with landmarks
        """
        try:
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")
            
            # Find face landmarks
            face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
            
            detections = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Convert to x, y, width, height format
                bbox = (left, top, right - left, bottom - top)
                
                # Extract landmarks if available
                landmarks = []
                if i < len(face_landmarks_list):
                    face_landmarks = face_landmarks_list[i]
                    # Flatten all landmark points
                    for feature_points in face_landmarks.values():
                        landmarks.extend(feature_points)
                
                # Create FaceDetection object
                detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    confidence=0.9,  # face_recognition doesn't provide confidence
                    pose=None  # Would need additional processing for pose
                )
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def get_largest_face(self, faces: List[FaceDetection]) -> Optional[FaceDetection]:
        """
        Get the largest detected face.
        
        Args:
            faces: List of face detections
            
        Returns:
            Largest face detection or None
        """
        if not faces:
            return None
        
        largest_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
        return largest_face
    
    def calculate_face_area_ratio(self, image_path: str, face_bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate the ratio of face area to total image area.
        
        Args:
            image_path: Path to the image
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Ratio of face area to image area (0.0 to 1.0)
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return 0.0
            
            image_area = image.shape[0] * image.shape[1]
            face_area = face_bbox[2] * face_bbox[3]
            
            return face_area / image_area if image_area > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def is_face_visible(self, face_detection: FaceDetection, min_visibility: float = 0.7) -> bool:
        """
        Check if face is sufficiently visible (not occluded).
        
        Args:
            face_detection: Face detection result
            min_visibility: Minimum visibility threshold
            
        Returns:
            True if face is sufficiently visible
        """
        # Simple heuristic: check if we have enough landmarks
        if len(face_detection.landmarks) < 10:
            return False
        
        # Check if bounding box is reasonable size
        bbox_area = face_detection.bbox[2] * face_detection.bbox[3]
        if bbox_area < 100 * 100:  # Less than 100x100 pixels
            return False
        
        return face_detection.confidence > min_visibility
    
    def estimate_pose(self, landmarks: List[Tuple[float, float]]) -> Optional[dict]:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: List of facial landmark points
            
        Returns:
            Dictionary with yaw, pitch, roll estimates or None
        """
        if len(landmarks) < 68:  # Need full 68-point landmarks
            return None
        
        try:
            # This is a simplified pose estimation
            # In production, you'd use a proper 3D head pose estimation model
            
            # Get key points
            nose_tip = landmarks[30] if len(landmarks) > 30 else landmarks[0]
            left_eye = landmarks[36] if len(landmarks) > 36 else landmarks[0]
            right_eye = landmarks[45] if len(landmarks) > 45 else landmarks[0]
            
            # Simple yaw estimation based on eye positions
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose_tip[0]
            
            # Rough yaw estimation (-90 to +90 degrees)
            yaw = (nose_x - eye_center_x) * 0.5  # Simplified calculation
            
            return {
                "yaw": max(-90, min(90, yaw)),
                "pitch": 0,  # Would need more complex calculation
                "roll": 0    # Would need more complex calculation
            }
            
        except Exception:
            return None
