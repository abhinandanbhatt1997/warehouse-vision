# 📦 Amazon Warehouse Computer Vision Toolkit

A computer vision-based toolkit using OpenCV to assist in warehouse operations like package dimension measurement, barcode scanning, and inventory monitoring.

## 🚀 Features

- 📏 **Package Dimension Measurement** (ArUco-based accurate real-world size estimation)
- 🔍 **Barcode & QR Scanner** (via `pyzbar`)
- 🧾 **Shelf Inventory Monitor** (color-based detection)
- 🖥️ Optimized for Lenovo T490s webcam (1280x720 resolution)

## 🖼️ Demo

![Demo GIF or Screenshot](assets/demo.gif)

## 🧠 Tech Stack

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Pyzbar
- ArUco Markers

## 📦 Installation

```bash
git clone https://github.com/abhinandanbhatt1997/warehouse-vision.git
cd amazon_warehouse_cv
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
