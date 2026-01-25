import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import time

class EmployeeTracker:
    def __init__(self, camera_id=0, idle_threshold_seconds=10):
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.idle_threshold = idle_threshold_seconds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.running = False
        
        # Initialize CPU-friendly HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Tracking data structures
        self.employees = {}  # {id: employee_data}
        self.next_employee_id = 1
        self.max_disappeared = 30  # frames before removing an employee
        
        # Heatmap setup
        self.heatmap_resolution = (64, 36)  # Low res for performance
        self.heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)
        self.frame_height = 720
        self.frame_width = 1280
        
    def _extract_features(self, frame, bbox):
        """Extract simple color histogram features for re-identification"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
            
        # Resize to standard size for consistency
        roi = cv2.resize(roi, (64, 128))
        
        # Extract color histogram (HSV space is better for clothing)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for box matching"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _match_employees(self, detections, frame):
        """Match current detections with tracked employees"""
        if len(self.employees) == 0:
            # No existing employees, create new ones
            for detection in detections:
                self._create_employee(detection, frame)
            return
        
        # Calculate IoU and feature similarity for all pairs
        matched = set()
        
        for detection in detections:
            best_match_id = None
            best_score = 0
            
            for emp_id, emp_data in self.employees.items():
                if emp_id in matched:
                    continue
                
                # IoU matching (spatial proximity)
                iou = self._calculate_iou(detection, emp_data['bbox'])
                
                # Feature similarity
                det_features = self._extract_features(frame, detection)
                if det_features is not None and emp_data['features'] is not None:
                    similarity = cv2.compareHist(
                        det_features.reshape(-1, 1),
                        emp_data['features'].reshape(-1, 1),
                        cv2.HISTCMP_CORREL
                    )
                else:
                    similarity = 0
                
                # Combined score (weighted)
                score = 0.7 * iou + 0.3 * max(0, similarity)
                
                if score > best_score and score > 0.3:  # Threshold
                    best_score = score
                    best_match_id = emp_id
            
            if best_match_id is not None:
                # Update existing employee
                self._update_employee(best_match_id, detection, frame)
                matched.add(best_match_id)
            else:
                # Create new employee
                self._create_employee(detection, frame)
        
        # Mark unmatched employees as disappeared
        for emp_id in list(self.employees.keys()):
            if emp_id not in matched:
                self.employees[emp_id]['disappeared'] += 1
                if self.employees[emp_id]['disappeared'] > self.max_disappeared:
                    del self.employees[emp_id]
    
    def _create_employee(self, bbox, frame):
        """Create a new tracked employee"""
        features = self._extract_features(frame, bbox)
        x, y, w, h = bbox
        
        self.employees[self.next_employee_id] = {
            'bbox': bbox,
            'features': features,
            'positions': deque(maxlen=30),  # Last 30 positions
            'timestamps': deque(maxlen=30),
            'disappeared': 0,
            'first_seen': time.time(),
            'color': tuple(np.random.randint(100, 255, 3).tolist())
        }
        
        # Store center position
        center = (x + w//2, y + h//2)
        self.employees[self.next_employee_id]['positions'].append(center)
        self.employees[self.next_employee_id]['timestamps'].append(time.time())
        
        self.next_employee_id += 1
    
    def _update_employee(self, emp_id, bbox, frame):
        """Update an existing employee's tracking data"""
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        self.employees[emp_id]['bbox'] = bbox
        self.employees[emp_id]['positions'].append(center)
        self.employees[emp_id]['timestamps'].append(time.time())
        self.employees[emp_id]['disappeared'] = 0
        
        # Update features periodically (every 10 frames)
        if len(self.employees[emp_id]['positions']) % 10 == 0:
            features = self._extract_features(frame, bbox)
            if features is not None:
                # Moving average of features
                old_features = self.employees[emp_id]['features']
                if old_features is not None:
                    self.employees[emp_id]['features'] = 0.7 * old_features + 0.3 * features
                else:
                    self.employees[emp_id]['features'] = features
    
    def _calculate_idle_time(self, emp_id):
        """Calculate how long an employee has been idle (in seconds)"""
        if emp_id not in self.employees:
            return 0
        
        positions = self.employees[emp_id]['positions']
        timestamps = self.employees[emp_id]['timestamps']
        
        if len(positions) < 5:
            return 0
        
        # Calculate movement in last N positions
        recent_positions = list(positions)[-15:]  # Last 15 positions (~0.5 seconds)
        
        # Calculate variance in position
        positions_array = np.array(recent_positions)
        variance = np.var(positions_array, axis=0).sum()
        
        # If variance is low, employee is idle
        if variance < 500:  # Threshold for "idle"
            # Calculate time since position stabilized
            current_time = timestamps[-1]
            for i in range(len(positions) - 1, 0, -1):
                if i >= len(timestamps):
                    continue
                pos_variance = np.var(list(positions)[max(0, i-5):i+1], axis=0).sum()
                if pos_variance > 500:
                    return current_time - timestamps[i]
            
            return current_time - timestamps[0]
        
        return 0
    
    def _update_heatmap(self):
        """Update the heatmap based on employee positions and idle times"""
        # Decay existing heatmap
        self.heatmap *= 0.95
        
        for emp_id, emp_data in self.employees.items():
            if len(emp_data['positions']) == 0:
                continue
            
            # Get current position
            x, y = emp_data['positions'][-1]
            
            # Convert to heatmap coordinates
            hmap_x = int((x / self.frame_width) * self.heatmap_resolution[0])
            hmap_y = int((y / self.frame_height) * self.heatmap_resolution[1])
            
            # Clamp to valid range
            hmap_x = max(0, min(hmap_x, self.heatmap_resolution[0] - 1))
            hmap_y = max(0, min(hmap_y, self.heatmap_resolution[1] - 1))
            
            # Calculate heat intensity based on idle time
            idle_time = self._calculate_idle_time(emp_id)
            intensity = min(1.0, idle_time / self.idle_threshold)
            
            # Add heat to heatmap with Gaussian spread
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = hmap_y + dy, hmap_x + dx
                    if 0 <= ny < self.heatmap_resolution[1] and 0 <= nx < self.heatmap_resolution[0]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        weight = np.exp(-dist / 2.0)
                        self.heatmap[ny, nx] = min(1.0, self.heatmap[ny, nx] + intensity * weight * 0.1)
    
    def _draw_heatmap_overlay(self, frame):
        """Draw heatmap overlay on frame"""
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(self.heatmap, (self.frame_width, self.frame_height))
        
        # Convert to color (green -> yellow -> red)
        heatmap_colored = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                intensity = heatmap_resized[i, j]
                if intensity < 0.5:
                    # Green to yellow
                    heatmap_colored[i, j] = [0, int(255 * (1 - intensity * 2)), int(255 * intensity * 2)]
                else:
                    # Yellow to red
                    heatmap_colored[i, j] = [0, int(255 * (2 - intensity * 2)), 255]
        
        # Blend with original frame
        mask = (heatmap_resized > 0.01).astype(np.uint8) * 255
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        frame = np.where(mask_3channel > 0, overlay, frame)
        
        return frame
    
    def run(self):
        """Main tracking loop"""
        self.running = True
        print("Employee Tracking System - Press 'q' to quit, 'h' to toggle heatmap")
        
        show_heatmap = True
        frame_skip = 2  # Process every 2nd frame for performance
        frame_count = 0
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Detect people (only every N frames for performance)
            if frame_count % frame_skip == 0:
                # Resize for faster detection
                small_frame = cv2.resize(frame, (640, 360))
                
                # Detect people
                boxes, weights = self.hog.detectMultiScale(
                    small_frame,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05
                )
                
                # Scale boxes back to original size
                detections = []
                for (x, y, w, h) in boxes:
                    x, y, w, h = x*2, y*2, w*2, h*2
                    detections.append((x, y, w, h))
                
                # Match and update employees
                self._match_employees(detections, frame)
                
                # Update heatmap
                self._update_heatmap()
            
            # Draw tracked employees
            for emp_id, emp_data in self.employees.items():
                x, y, w, h = emp_data['bbox']
                color = emp_data['color']
                
                # Calculate idle time
                idle_time = self._calculate_idle_time(emp_id)
                
                # Change box color based on idle status
                if idle_time > self.idle_threshold:
                    box_color = (0, 0, 255)  # Red for idle
                    status = "IDLE"
                elif idle_time > self.idle_threshold * 0.5:
                    box_color = (0, 165, 255)  # Orange for warning
                    status = "WARNING"
                else:
                    box_color = (0, 255, 0)  # Green for active
                    status = "ACTIVE"
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                
                # Draw ID and status
                label = f"EMP-{emp_id:03d} | {status}"
                cv2.putText(display_frame, label, (x, y-10), 
                           self.font, 0.5, box_color, 2)
                
                # Draw idle time if applicable
                if idle_time > 2:
                    time_label = f"Idle: {idle_time:.1f}s"
                    cv2.putText(display_frame, time_label, (x, y+h+20),
                               self.font, 0.4, box_color, 1)
                
                # Draw trajectory
                if len(emp_data['positions']) > 1:
                    points = np.array(list(emp_data['positions']), dtype=np.int32)
                    cv2.polylines(display_frame, [points], False, color, 1)
            
            # Draw heatmap overlay
            if show_heatmap:
                display_frame = self._draw_heatmap_overlay(display_frame)
            
            # Draw info panel
            info_text = f"Employees: {len(self.employees)} | Frame: {frame_count}"
            cv2.putText(display_frame, info_text, (10, 30),
                       self.font, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Employee Tracking System", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('h'):
                show_heatmap = not show_heatmap
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = EmployeeTracker(idle_threshold_seconds=10)
    tracker.run()
