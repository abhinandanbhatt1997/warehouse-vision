import cv2
import json
import os
from datetime import datetime

class AccuracyValidator:
    """
    Tools to validate employee tracking accuracy by comparing
    against ground truth annotations
    """
    
    def __init__(self, video_path, insights_json_path):
        self.video_path = video_path
        self.insights_json_path = insights_json_path
        
        # Load insights data
        with open(insights_json_path, 'r') as f:
            self.insights = json.load(f)
        
        # Ground truth annotations (manual verification)
        self.ground_truth = {
            "total_people": 0,
            "idle_events": [],
            "annotations": {}
        }
        
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame = 0
        self.paused = False
        
    def annotate_ground_truth(self):
        """
        Interactive tool to create ground truth annotations
        Manual verification by stepping through video
        """
        print("="*60)
        print("GROUND TRUTH ANNOTATION TOOL")
        print("="*60)
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  'p' - Mark person detected at current frame")
        print("  'i' - Mark idle event start")
        print("  'a' - Mark active (end idle)")
        print("  's' - Save ground truth")
        print("  'q' - Quit")
        print("  LEFT/RIGHT arrow - Step frame by frame when paused")
        print("="*60)
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video")
                    break
                self.current_frame += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            # Display frame with current annotations
            display = frame.copy()
            
            # Show current frame info
            cv2.putText(display, f"Frame: {self.current_frame}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show annotation count
            frame_annotations = self.ground_truth["annotations"].get(self.current_frame, {})
            cv2.putText(display, f"People marked: {frame_annotations.get('people_count', 0)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.paused:
                cv2.putText(display, "PAUSED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Ground Truth Annotation", display)
            
            key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('p'):
                # Mark person at current frame
                if self.current_frame not in self.ground_truth["annotations"]:
                    self.ground_truth["annotations"][self.current_frame] = {"people_count": 0, "idle": False}
                self.ground_truth["annotations"][self.current_frame]["people_count"] += 1
                print(f"Frame {self.current_frame}: Marked person (total: {self.ground_truth['annotations'][self.current_frame]['people_count']})")
            elif key == ord('i'):
                # Mark idle event
                if self.current_frame not in self.ground_truth["annotations"]:
                    self.ground_truth["annotations"][self.current_frame] = {"people_count": 0, "idle": False}
                self.ground_truth["annotations"][self.current_frame]["idle"] = True
                print(f"Frame {self.current_frame}: Marked as IDLE")
            elif key == ord('a'):
                # Mark active
                if self.current_frame not in self.ground_truth["annotations"]:
                    self.ground_truth["annotations"][self.current_frame] = {"people_count": 0, "idle": False}
                self.ground_truth["annotations"][self.current_frame]["idle"] = False
                print(f"Frame {self.current_frame}: Marked as ACTIVE")
            elif key == ord('s'):
                self.save_ground_truth()
            elif key == 81:  # Left arrow
                if self.paused:
                    self.current_frame = max(0, self.current_frame - 1)
            elif key == 83:  # Right arrow
                if self.paused:
                    self.current_frame += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save before exit
        self.save_ground_truth()
    
    def save_ground_truth(self):
        """Save ground truth annotations to JSON"""
        output_path = self.insights_json_path.replace(".json", "_ground_truth.json")
        with open(output_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        print(f"✅ Ground truth saved to: {output_path}")
    
    def calculate_accuracy(self, ground_truth_path):
        """
        Calculate accuracy metrics by comparing insights with ground truth
        """
        print("\n" + "="*60)
        print("ACCURACY ANALYSIS")
        print("="*60)
        
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            gt = json.load(f)
        
        # Detection accuracy
        print("\n📊 Detection Accuracy:")
        
        total_gt_people = sum(ann.get("people_count", 0) for ann in gt["annotations"].values())
        total_detected = len(self.insights["employees"])
        
        if total_gt_people > 0:
            detection_rate = (total_detected / total_gt_people) * 100
            print(f"   Ground Truth People: {total_gt_people}")
            print(f"   Detected People: {total_detected}")
            print(f"   Detection Rate: {detection_rate:.1f}%")
        
        # Idle detection accuracy
        print("\n⏱️  Idle Detection Accuracy:")
        
        gt_idle_frames = [frame for frame, data in gt["annotations"].items() 
                         if data.get("idle", False)]
        detected_idle_events = self.insights["summary"].get("total_idle_events", 0)
        
        print(f"   Ground Truth Idle Frames: {len(gt_idle_frames)}")
        print(f"   Detected Idle Events: {detected_idle_events}")
        
        # Frame-by-frame comparison
        print("\n🎯 Frame-by-Frame Analysis:")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for frame, gt_data in gt["annotations"].items():
            gt_count = gt_data.get("people_count", 0)
            # Simplified: compare with average detection in that time window
            # In production, you'd match frame numbers precisely
            if gt_count > 0 and total_detected > 0:
                true_positives += 1
            elif gt_count > 0 and total_detected == 0:
                false_negatives += 1
            elif gt_count == 0 and total_detected > 0:
                false_positives += 1
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"   Precision: {precision:.2f}")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"   Recall: {recall:.2f}")
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"   F1 Score: {f1_score:.2f}")
        
        print("\n" + "="*60)
    
    def visual_comparison(self):
        """
        Play analyzed video side-by-side with original
        to visually verify accuracy
        """
        print("\n🎬 Visual Comparison Tool")
        print("Playing analyzed video. Check for:")
        print("  ✓ Correct bounding boxes around people")
        print("  ✓ Persistent IDs (same person = same ID)")
        print("  ✓ Accurate idle detection")
        print("  ✓ No false detections")
        
        # Find analyzed video
        session_id = self.insights["session_id"]
        analyzed_video = f"output_videos/analyzed_{session_id}.mp4"
        
        if not os.path.exists(analyzed_video):
            print(f"❌ Analyzed video not found: {analyzed_video}")
            return
        
        cap = cv2.VideoCapture(analyzed_video)
        
        print("\nPress 'q' to quit, SPACE to pause")
        
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            cv2.imshow("Analyzed Video - Check Accuracy", frame)
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()


def quick_accuracy_check(insights_json_path):
    """
    Quick automated checks on insights data
    """
    print("="*60)
    print("QUICK ACCURACY CHECKS")
    print("="*60)
    
    with open(insights_json_path, 'r') as f:
        insights = json.load(f)
    
    # Check 1: Reasonable employee count
    print("\n✓ Employee Count Check:")
    emp_count = insights["summary"]["total_employees_tracked"]
    print(f"   Total employees tracked: {emp_count}")
    if emp_count == 0:
        print("   ⚠️  WARNING: No employees detected!")
    elif emp_count > 50:
        print("   ⚠️  WARNING: Unusually high employee count - possible false detections")
    else:
        print("   ✓ Reasonable count")
    
    # Check 2: Idle event ratio
    print("\n✓ Idle Event Ratio Check:")
    idle_events = insights["summary"]["total_idle_events"]
    duration = insights["summary"]["session_duration_seconds"]
    
    if emp_count > 0:
        idle_rate = idle_events / emp_count
        print(f"   Idle events per employee: {idle_rate:.1f}")
        if idle_rate > 10:
            print("   ⚠️  WARNING: Very high idle rate - check threshold settings")
        else:
            print("   ✓ Normal idle rate")
    
    # Check 3: Event timeline consistency
    print("\n✓ Event Timeline Check:")
    events = insights["events"]
    print(f"   Total events logged: {len(events)}")
    
    status_changes = [e for e in events if e["type"] == "status_change"]
    print(f"   Status change events: {len(status_changes)}")
    
    if len(status_changes) > 0:
        print("   ✓ Events being logged")
    else:
        print("   ⚠️  WARNING: No status changes detected")
    
    # Check 4: Data completeness
    print("\n✓ Data Completeness Check:")
    required_fields = ["session_id", "start_time", "summary", "employees", "events"]
    missing = [f for f in required_fields if f not in insights]
    
    if missing:
        print(f"   ❌ Missing fields: {missing}")
    else:
        print("   ✓ All required fields present")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python accuracy_validator.py <video_path> <insights_json>")
        print("\nOr for quick check:")
        print("  python accuracy_validator.py quick <insights_json>")
        sys.exit(1)
    
    if sys.argv[1] == "quick":
        quick_accuracy_check(sys.argv[2])
    else:
        video_path = sys.argv[1]
        insights_json = sys.argv[2]
        
        validator = AccuracyValidator(video_path, insights_json)
        
        print("\nSelect mode:")
        print("1. Create ground truth annotations")
        print("2. Calculate accuracy (requires ground truth)")
        print("3. Visual comparison")
        print("4. Quick automated checks")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            validator.annotate_ground_truth()
        elif choice == "2":
            gt_path = insights_json.replace(".json", "_ground_truth.json")
            if os.path.exists(gt_path):
                validator.calculate_accuracy(gt_path)
            else:
                print(f"❌ Ground truth not found: {gt_path}")
                print("   Run mode 1 first to create ground truth")
        elif choice == "3":
            validator.visual_comparison()
        elif choice == "4":
            quick_accuracy_check(insights_json)
