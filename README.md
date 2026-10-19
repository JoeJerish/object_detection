# Vision Assist AI — Real-Time Object & Mask Detection Assistant

A modern, multi-threaded real-time vision desktop application with object detection (YOLOv8), face-mask classification, and an intelligent Windows SAPI5 female voice assistant.

---

## 🌟 Key Features

1. **Real-Time Object Detection**:
   - Detects multiple objects simultaneously (Person, Car, Laptop, Bottle, Phone, Chair, Backpack, Dog, Cat, etc.).
   - Uses lightweight **YOLOv8n** for maximum speed on standard laptop CPUs.
   - Real-time bounding boxes with class labels and confidence percentages.

2. **Face & Mask Detection**:
   - Detects faces and classifies mask status (**MASK ON** vs **MASK OFF**).
   - PyTorch MobileNetV2 neural network face mask classification.
   - Temporal stability filter to eliminate flickering between MASK ON and MASK OFF.
   - Color-coded face bounding boxes (Green = Mask ON, Red = Mask OFF).

3. **High-Quality Lag-Free Camera Pipeline**:
   - Multi-threaded asynchronous video acquisition (`WebcamStream`).
   - Auto-probes maximum supported resolution (**1080p / 720p @ 30 FPS**).
   - Dynamic lighting enhancement (CLAHE).
   - Dedicated AI inference thread so camera preview stays smooth without freezing.

4. **Intelligent Windows Voice Assistant**:
   - Uses native Windows **SAPI5** text-to-speech.
   - Auto-selects **Microsoft Zira Desktop** (Female English voice).
   - Dedicated non-blocking speech queue thread.
   - Intelligent event cooldowns to eliminate repetitive announcements.
   - Speaks mask state changes ("Person detected without a mask." / "Person detected wearing a mask.").
   - Initial startup greeting: *"Vision Assist AI is ready."*

5. **Modern CustomTkinter Dashboard**:
   - Dark theme dashboard displaying live camera resolution, live FPS, object counts, person counts, and detailed mask status cards.
   - Real-time control sliders for Confidence Threshold (10%–90%), Voice Assistant Toggle, Voice Speed (WPM), and Voice Volume (0%–100%).

---

## 📋 Requirements & Dependencies

- **OS**: Windows 10 / Windows 11
- **Python**: Python 3.9+ (tested on Python 3.14)
- **Webcam**: Laptop built-in webcam or USB webcam

### Required Python Packages:
```bash
pip install opencv-python ultralytics torch torchvision pyttsx3 customtkinter pillow pypiwin32
```

---

## 🚀 Quick Start Guide

### 1. Clone / Open Directory
Open terminal or PowerShell in the application folder:
```powershell
cd "c:\Users\joeje\Downloads\face mask detection"
```

### 2. Install Dependencies
Run the pip install command:
```powershell
pip install opencv-python ultralytics torch torchvision pyttsx3 customtkinter pillow pypiwin32
```

### 3. Model Files
- **YOLOv8n**: Automatically downloaded on first launch to `./yolov8n.pt` (~6.2 MB).
- **Face Cascade**: Included automatically with OpenCV (`cv2.data.haarcascades`).

### 4. Run the Application
```powershell
python main.py
```

---

## 🔊 Windows Female Voice Setup Instructions

The application automatically searches for Windows female SAPI5 voices:
1. `Microsoft Zira Desktop - English (United States)`
2. `Microsoft Eva Mobile`
3. `Microsoft Hazel Desktop`

If a female voice is not installed on your Windows machine, follow these steps to add Microsoft Zira:

1. Open **Windows Settings** (`Win + I`).
2. Go to **Time & Language** ➔ **Speech**.
3. Under **Manage voices**, click **Add voices**.
4. Search for **English (United States)** or **English (United Kingdom)** and check the box.
5. Click **Add**. Windows will download `Microsoft Zira` / `Microsoft Hazel`.
6. Restart the application (`python main.py`).

---

## ⚙️ Voice Speed & Volume Controls

Inside the application dashboard panel on the right:
- **Detection Confidence**: Drag slider from `10%` to `90%` to filter out uncertain detections.
- **Voice Assistant Toggle**: Turn spoken alerts ON or OFF anytime.
- **Voice Speed (WPM)**: Adjust speech speed from `100 WPM` (slow) up to `240 WPM` (fast). Default is `165 WPM`.
- **Voice Volume**: Adjust output volume from `0%` (mute) to `100%`.

---

## 📁 File Structure

```
face mask detection/
├── camera_stream.py      # Threaded webcam capture & resolution probing
├── object_detector.py    # YOLOv8 object & person detection wrapper
├── face_mask_detector.py # OpenCV & PyTorch MobileNetV2 Face Mask Classifier
├── voice_assistant.py    # Threaded Windows SAPI5 TTS engine & event queue
├── app_gui.py            # CustomTkinter dashboard & video canvas
├── main.py               # Main application entry point
└── README.md             # Project documentation & setup instructions
```
