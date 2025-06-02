import cv2
import numpy as np

class PackageMeasurer:
    def __init__(self, camera_id=0, marker_length_cm=4.0):
        self.camera = cv2.VideoCapture(camera_id)
        self.marker_length_cm = marker_length_cm
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.running = False

        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        self.running = True
        print("Package Measurement with ArUco Marker - Press 'q' to quit")

        # Updated ArUco detection setup
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  # New detector object

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Updated detection call
            corners, ids, _ = detector.detectMarkers(gray)  # Use the detector object

            pixels_per_cm = None

            if ids is not None and len(ids) > 0:
                # Use the first detected marker
                marker_corners = corners[0][0]
                marker_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
                pixels_per_cm = marker_width_px / self.marker_length_cm

                cv2.polylines(frame, [np.int32(marker_corners)], True, (0, 255, 255), 2)
                cv2.putText(frame, "ArUco Marker Detected", (10, 30), self.font, 0.7, (0, 255, 255), 2)

            # Rest of your package detection code remains the same...
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                width_px = min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
                height_px = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))

                if pixels_per_cm:
                    width_cm = width_px / pixels_per_cm
                    height_cm = height_px / pixels_per_cm

                    cv2.putText(frame, f"W: {width_cm:.1f} cm", (box[0][0], box[0][1] - 25), self.font, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"H: {height_cm:.1f} cm", (box[0][0], box[0][1] - 5), self.font, 0.6, (255, 255, 0), 2)

                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

            cv2.imshow("Dimension Measurement", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    measurer = PackageMeasurer()
    measurer.run()
