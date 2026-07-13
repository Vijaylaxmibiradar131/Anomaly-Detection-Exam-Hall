"""
Privacy Enhancement Module
Implements face blurring and anonymization for privacy protection
"""

import cv2
import numpy as np
from pathlib import Path

class PrivacyEnhancer:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    
    def pixelate_faces(self, image, pixel_size=20):
        """
        Detect and pixelate faces in an image
        
        Args:
            image: Input image (numpy array)
            pixel_size: Size of pixelation blocks
        
        Returns:
            Image with pixelated faces
        """
        if image is None:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Create a copy of the image
        pixelated_image = image.copy()
        
        # Pixelate each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = pixelated_image[y:y+h, x:x+w]
            
            # Resize down and up to create pixelation effect
            small = cv2.resize(face_region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Replace face region with pixelated version
            pixelated_image[y:y+h, x:x+w] = pixelated
        
        return pixelated_image
    
    def add_black_boxes(self, image):
        """
        Detect and cover faces with black boxes
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Image with black boxes over faces
        """
        if image is None:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Create a copy of the image
        boxed_image = image.copy()
        
        # Draw black rectangle over each face
        for (x, y, w, h) in faces:
            cv2.rectangle(boxed_image, (x, y), (x+w, y+h), (0, 0, 0), -1)
        
        return boxed_image
    
    def process_snapshot_directory(self, snapshot_dir, output_dir=None, method='blur', blur_factor=50, pixel_size=20):
        """
        Process all snapshots in a directory to apply privacy enhancement
        
        Args:
            snapshot_dir: Directory containing snapshots
            output_dir: Output directory (if None, overwrites original)
            method: 'pixelate' or 'black_box'
            pixel_size: Pixel size for pixelate method
        
        Returns:
            Number of processed images
        """
        snapshot_path = Path(snapshot_dir)
        if not snapshot_path.exists():
            print(f"Snapshot directory not found: {snapshot_dir}")
            return 0
        
        # Set output directory
        if output_dir is None:
            output_path = snapshot_path
        else:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        # Process each image
        processed_count = 0
        image_files = list(snapshot_path.glob('*.jpg')) + list(snapshot_path.glob('*.png'))
        
        for img_file in image_files:
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Apply privacy enhancement
            if method == 'pixelate':
                processed_image = self.pixelate_faces(image, pixel_size)
            elif method == 'black_box':
                processed_image = self.add_black_boxes(image)
            else:
                print(f"Unknown method: {method}")
                continue
            
            # Save processed image
            if output_dir is None:
                output_file = img_file
            else:
                output_file = output_path / img_file.name
            
            cv2.imwrite(str(output_file), processed_image)
            processed_count += 1
        
        print(f"Processed {processed_count} images with {method} method")
        return processed_count
    
    def process_video_with_privacy(self, video_path, output_path, method='blur', blur_factor=50):
        """
        Process a video and apply privacy enhancement
        
        Args:
            video_path: Input video path
            output_path: Output video path
            method: Privacy method to apply
        
        Returns:
            Success status and message
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Could not open video: {video_path}"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply privacy enhancement
            if method == 'pixelate':
                processed_frame = self.pixelate_faces(frame)
            elif method == 'black_box':
                processed_frame = self.add_black_boxes(frame)
            else:
                processed_frame = frame
            
            out.write(processed_frame)
            processed_frames += 1
            
            # Progress indication
            if processed_frames % 30 == 0:
                progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing video: {progress:.1f}% complete", end='\r')
        
        cap.release()
        out.release()
        
        print(f"\nProcessed {processed_frames} frames")
        return True, f"Privacy-enhanced video saved to {output_path}"
