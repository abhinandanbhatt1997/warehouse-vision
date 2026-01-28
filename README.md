# 📦 Amazon Warehouse Computer Vision Toolkit

A comprehensive computer vision toolkit using OpenCV for warehouse operations including package dimension measurement, employee tracking with heatmap analytics, barcode scanning, and inventory monitoring.

## 🚀 Features

### Package Management
- 📏 **Package Dimension Measurement** - ArUco marker-based accurate real-world size estimation
- 🔍 **Barcode & QR Scanner** - Automated scanning via `pyzbar`
- 🧾 **Shelf Inventory Monitor** - Color-based detection system

### Employee Tracking & Analytics
- 👥 **Real-time Employee Tracking** - CPU-optimized person detection with persistent ID assignment
- 🎯 **Internal Identifier System** - No ArUco markers needed, uses visual appearance matching
- 🗺️ **Activity Heatmap** - Visual representation of employee movement and dwell time
- ⚠️ **Idle Detection** - Automatic flagging when employees remain stationary too long
- 📊 **Status Indicators** - Green (Active) → Orange (Warning) → Red (Idle)

## 🖼️ Demo

![Package Measurement Demo](assets/demo.gif)
![Employee Tracking Demo](assets/employee_tracking.gif)

## 🧠 Tech Stack

- **Python** 3.8+
- **OpenCV** (cv2) - Computer vision operations
- **NumPy** - Numerical computations
- **Pyzbar** - Barcode/QR code detection
- **HOG Detector** - CPU-friendly person detection

## 💻 System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Intel i5 or equivalent (optimized for Lenovo ThinkPad T490s)
- **Webcam**: 720p minimum (1280x720), tested with Lenovo T490s webcam
- **No GPU required** - All models are CPU-optimized

## 📦 Installation

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/abhinandanbhatt1997/warehouse-vision.git
cd warehouse-vision

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install camera utilities (Linux only)
sudo apt install v4l-utils  # Ubuntu/Debian
```

### Windows

```bash
# Clone the repository
git clone https://github.com/abhinandanbhatt1997/warehouse-vision.git
cd warehouse-vision

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### WSL Users

**Note**: WSL doesn't have native camera access. For best results, use Windows Python or set up USB/IP passthrough. Alternatively, test with video files.

## 🎮 Usage

### Package Dimension Measurement

```bash
python warehouse-vision.py
```

**Controls:**
- `q` - Quit application

**Requirements:**
- Print a 4cm x 4cm ArUco marker (DICT_5X5_100)
- Place marker in camera view for calibration
- Position packages near marker for accurate measurements

### Employee Tracking System

```bash
python employee_tracker.py
```

**Controls:**
- `q` - Quit application
- `h` - Toggle heatmap overlay

**Configuration:**
```python
# Adjust idle detection threshold (default: 10 seconds)
tracker = EmployeeTracker(idle_threshold_seconds=10)

# Change camera index if needed
tracker = EmployeeTracker(camera_id=0)
```

## 🔧 Troubleshooting

### Camera Not Detected

**Check available cameras:**
```bash
# Linux
ls -la /dev/video*
v4l2-ctl --list-devices

# Test different camera indices
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

**Common fixes:**
```bash
# Linux - Add user to video group
sudo usermod -a -G video $USER
# Log out and log back in

# Check if camera is busy
sudo lsof /dev/video0
```

### Performance Issues

If tracking is slow on your system:

1. **Reduce frame skip** in `employee_tracker.py`:
```python
frame_skip = 3  # Process every 3rd frame instead of 2nd
```

2. **Reduce detection resolution**:
```python
small_frame = cv2.resize(frame, (480, 270))  # Instead of 640x360
```

3. **Lower camera resolution**:
```python
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Testing with Video Files

For testing without a camera:

```python
# Modify __init__ method
self.camera = cv2.VideoCapture('test_warehouse.mp4')
```

## 📊 Understanding the Heatmap

The activity heatmap uses a color gradient to show employee activity:

- **Green zones** - High movement, active work areas
- **Yellow zones** - Moderate activity or recent movement
- **Red zones** - Prolonged idle time (exceeds threshold)

Heat intensity builds up when employees remain stationary and decays over time as they move.

## 🎯 How Employee Tracking Works

1. **Detection** - HOG (Histogram of Oriented Gradients) detector identifies people in frame
2. **Feature Extraction** - Color histograms in HSV space for appearance matching
3. **ID Assignment** - Combines spatial proximity (IoU) and visual similarity
4. **Tracking** - Maintains trajectory history and calculates movement variance
5. **Idle Detection** - Flags employees with position variance below threshold
6. **Heatmap Generation** - Accumulates dwell time on spatial grid with Gaussian smoothing

## 🔐 Privacy Considerations

- Employee IDs are session-based and not permanently stored
- No facial recognition is performed
- Visual features are simple color histograms, not biometric data
- System designed for activity monitoring, not individual surveillance

## 📝 Configuration Files

### requirements.txt
```
opencv-python
numpy
pyzbar
python-dotenv
```

### .env (optional)
```
CAMERA_INDEX=0
IDLE_THRESHOLD_SECONDS=10
FRAME_SKIP=2
```

## 🛠️ Project Structure

```
warehouse-vision/
├── README.md
├── requirements.txt
├── warehouse-vision.py      # Package measurement module
├── employee_tracker.py      # Employee tracking module
├── assets/
│   ├── demo.gif
│   └── employee_tracking.gif
└── myenv/                   # Virtual environment (not committed)
```

## 🚧 Roadmap

- [ ] Integrate package measurement and employee tracking into single interface
- [ ] Add barcode scanning module
- [ ] Export heatmap analytics to CSV/JSON
- [ ] Multi-camera support for larger warehouses
- [ ] Historical tracking data visualization
- [ ] Alert system for prolonged idle detection
- [ ] Zone-based analytics (loading, packing, storage areas)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Abhinandan Bhatt**
- GitHub: [@abhinandanbhatt1997](https://github.com/abhinandanbhatt1997)

## 🙏 Acknowledgments

- OpenCV community for excellent computer vision libraries
- ArUco marker detection for accurate measurements
- HOG descriptor paper by Dalal and Triggs

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: [abhinandanbhatt1997@gmail.com]

## ⚡ Performance Benchmarks

Tested on **Lenovo ThinkPad T490s** (Intel i7-8565U, 16GB RAM):

| Module | Resolution | FPS | CPU Usage |
|--------|-----------|-----|-----------|
| Package Measurement | 1280x720 | 25-30 | ~15% |
| Employee Tracking | 1280x720 | 15-20 | ~25% |
| Combined Mode | 1280x720 | 12-15 | ~30% |

---

**Made with ❤️ for warehouse automation**
