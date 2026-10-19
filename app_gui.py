import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import time
import threading

from camera_stream import WebcamStream
from object_detector import ObjectDetector
from face_mask_detector import FaceMaskDetector
from voice_assistant import VoiceAssistant


# Set CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VisionAssistApp(ctk.CTk):
    """
    Main GUI application for Real-Time AI Vision & Mask Detection Assistant.
    """

    def __init__(self):
        super().__init__()

        self.title("Vision Assist AI - Object & Mask Detector")
        self.geometry("1280x800")
        self.minsize(1024, 680)

        # Core Components
        self.webcam = None
        self.object_detector = None
        self.mask_detector = None
        self.voice_assistant = None

        # State Variables
        self.is_running = False
        self.latest_frame = None
        self.latest_detections = {'detections': [], 'person_boxes': [], 'summary_counts': {}}
        self.latest_mask_results = []
        self.ai_lock = threading.Lock()

        # UI Variables
        self.conf_var = ctk.DoubleVar(value=0.40)
        self.voice_enabled_var = ctk.BooleanVar(value=True)
        self.voice_rate_var = ctk.IntVar(value=165)
        self.voice_vol_var = ctk.DoubleVar(value=1.0)

        # Build UI layout
        self._create_widgets()

        # Initialize AI subsystems in background thread
        self.after(100, self._async_init_subsystems)

    def _create_widgets(self):
        """Constructs modern CustomTkinter layout."""
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT PANEL: Video Display ----------------
        self.left_panel = ctk.CTkFrame(self, corner_radius=12)
        self.left_panel.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=0)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # App Header
        self.header_label = ctk.CTkLabel(
            self.left_panel,
            text="👁️ VISION ASSIST AI — REAL-TIME CAMERA PREVIEW",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.header_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Video Canvas Frame
        self.video_frame = ctk.CTkFrame(self.left_panel, fg_color="#121212", corner_radius=8)
        self.video_frame.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Initializing Camera & AI Models...\nPlease Wait.",
            font=ctk.CTkFont(size=16)
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ---------------- RIGHT PANEL: Dashboard Controls ----------------
        self.right_panel = ctk.CTkFrame(self, width=350, corner_radius=12)
        self.right_panel.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="nsew")
        self.right_panel.grid_rowconfigure(5, weight=1)

        # 1. System Status Card
        self.status_card = ctk.CTkFrame(self.right_panel, corner_radius=8)
        self.status_card.pack(fill="x", padx=15, pady=(15, 10))

        self.status_title = ctk.CTkLabel(self.status_card, text="SYSTEM STATUS", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_title.pack(anchor="w", padx=10, pady=(8, 2))

        self.cam_status_lbl = ctk.CTkLabel(self.status_card, text="Camera: OFF", font=ctk.CTkFont(size=12), text_color="#FF5555")
        self.cam_status_lbl.pack(anchor="w", padx=10, pady=2)

        self.res_lbl = ctk.CTkLabel(self.status_card, text="Resolution: N/A", font=ctk.CTkFont(size=12))
        self.res_lbl.pack(anchor="w", padx=10, pady=2)

        self.fps_lbl = ctk.CTkLabel(self.status_card, text="FPS: 0.0", font=ctk.CTkFont(size=12))
        self.fps_lbl.pack(anchor="w", padx=10, pady=(2, 8))

        # 2. Live Detections Card
        self.det_card = ctk.CTkFrame(self.right_panel, corner_radius=8)
        self.det_card.pack(fill="x", padx=15, pady=10)

        self.det_title = ctk.CTkLabel(self.det_card, text="DETECTION SUMMARY", font=ctk.CTkFont(size=14, weight="bold"))
        self.det_title.pack(anchor="w", padx=10, pady=(8, 2))

        self.total_obj_lbl = ctk.CTkLabel(self.det_card, text="Objects Detected: 0", font=ctk.CTkFont(size=13))
        self.total_obj_lbl.pack(anchor="w", padx=10, pady=2)

        self.persons_lbl = ctk.CTkLabel(self.det_card, text="Persons Detected: 0", font=ctk.CTkFont(size=13))
        self.persons_lbl.pack(anchor="w", padx=10, pady=2)

        # Mask Details Frame
        self.mask_scroll = ctk.CTkScrollableFrame(self.det_card, height=110, fg_color="#1E1E1E")
        self.mask_scroll.pack(fill="x", padx=10, pady=(4, 8))

        self.mask_info_lbl = ctk.CTkLabel(self.mask_scroll, text="No person detected", font=ctk.CTkFont(size=12), text_color="#888888")
        self.mask_info_lbl.pack(anchor="w")

        # 3. Controls Card
        self.ctrl_card = ctk.CTkFrame(self.right_panel, corner_radius=8)
        self.ctrl_card.pack(fill="x", padx=15, pady=10)

        self.ctrl_title = ctk.CTkLabel(self.ctrl_card, text="CONTROLS & SETTINGS", font=ctk.CTkFont(size=14, weight="bold"))
        self.ctrl_title.pack(anchor="w", padx=10, pady=(8, 5))

        # Confidence Slider
        self.conf_title = ctk.CTkLabel(self.ctrl_card, text="Detection Confidence: 40%", font=ctk.CTkFont(size=12))
        self.conf_title.pack(anchor="w", padx=10, pady=(2, 0))

        self.conf_slider = ctk.CTkSlider(
            self.ctrl_card,
            from_=0.10,
            to=0.90,
            number_of_steps=16,
            variable=self.conf_var,
            command=self._on_conf_change
        )
        self.conf_slider.pack(fill="x", padx=10, pady=(0, 8))

        # Voice Assistant Toggle
        self.voice_switch = ctk.CTkSwitch(
            self.ctrl_card,
            text="Voice Assistant",
            variable=self.voice_enabled_var,
            command=self._on_voice_toggle
        )
        self.voice_switch.pack(anchor="w", padx=10, pady=5)

        # Voice Speed Slider
        self.speed_title = ctk.CTkLabel(self.ctrl_card, text="Voice Speed (WPM): 165", font=ctk.CTkFont(size=12))
        self.speed_title.pack(anchor="w", padx=10, pady=(5, 0))

        self.speed_slider = ctk.CTkSlider(
            self.ctrl_card,
            from_=100,
            to=240,
            number_of_steps=14,
            variable=self.voice_rate_var,
            command=self._on_speed_change
        )
        self.speed_slider.pack(fill="x", padx=10, pady=(0, 8))

        # Voice Volume Slider
        self.vol_title = ctk.CTkLabel(self.ctrl_card, text="Voice Volume: 100%", font=ctk.CTkFont(size=12))
        self.vol_title.pack(anchor="w", padx=10, pady=(5, 0))

        self.vol_slider = ctk.CTkSlider(
            self.ctrl_card,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=self.voice_vol_var,
            command=self._on_volume_change
        )
        self.vol_slider.pack(fill="x", padx=10, pady=(0, 10))

        # 4. Action Buttons
        self.btn_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=15, pady=(10, 15))

        self.start_btn = ctk.CTkButton(
            self.btn_frame,
            text="▶ START CAMERA",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#28A745",
            hover_color="#218838",
            height=40,
            command=self.start_camera
        )
        self.start_btn.pack(fill="x", pady=5)

        self.stop_btn = ctk.CTkButton(
            self.btn_frame,
            text="⏹ STOP CAMERA",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#DC3545",
            hover_color="#C82333",
            height=40,
            state="disabled",
            command=self.stop_camera
        )
        self.stop_btn.pack(fill="x", pady=5)

    def _async_init_subsystems(self):
        """Asynchronously loads AI models and initializes voice assistant."""
        def load_task():
            try:
                # 1. Voice assistant init
                self.voice_assistant = VoiceAssistant(
                    rate=self.voice_rate_var.get(),
                    volume=self.voice_vol_var.get()
                )

                # 2. Object detector init
                self.object_detector = ObjectDetector(conf_threshold=self.conf_var.get())

                # 3. Face & Mask detector init
                self.mask_detector = FaceMaskDetector()

                # Notify GUI thread initialization complete
                self.after(0, self._on_subsystems_ready)

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: messagebox.showerror("Initialization Error", f"Failed to load AI models:\n{err_msg}"))

        threading.Thread(target=load_task, daemon=True).start()

    def _on_subsystems_ready(self):
        """Called when models are loaded."""
        self.video_label.configure(text="AI Vision Assistant Ready.\nClick 'START CAMERA' to begin.")
        # Trigger initial startup voice greeting
        if self.voice_assistant:
            self.voice_assistant.speak("Vision Assist AI is ready.", priority=1, force=True)

    def start_camera(self):
        """Starts the webcam and detection threads."""
        if self.is_running:
            return

        try:
            self.webcam = WebcamStream(target_width=1280, target_height=720, target_fps=30)
            self.webcam.start()

            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.cam_status_lbl.configure(text="Camera: ON", text_color="#28A745")

            w, h = self.webcam.get_resolution()
            self.res_lbl.configure(text=f"Resolution: {w}x{h}")

            # Start background AI worker thread
            self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
            self.ai_thread.start()

            # Start GUI frame rendering loop
            self._render_gui_loop()

        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not start webcam:\n{str(e)}")
            self.stop_camera()

    def stop_camera(self):
        """Stops video stream and AI processing."""
        self.is_running = False
        if self.webcam:
            self.webcam.stop()
            self.webcam = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.cam_status_lbl.configure(text="Camera: OFF", text_color="#FF5555")
        self.fps_lbl.configure(text="FPS: 0.0")
        self.video_label.configure(image="", text="Camera Stopped.")

    def _ai_worker_loop(self):
        """Background thread executing YOLO & Face Mask inference."""
        while self.is_running:
            if self.webcam is None:
                break

            grabbed, frame = self.webcam.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Run Object Detector
            if self.object_detector:
                det_results = self.object_detector.detect(frame)
            else:
                det_results = {'detections': [], 'person_boxes': [], 'summary_counts': {}}

            # Run Face & Mask Detector on person crops
            mask_results = []
            if self.mask_detector and det_results['person_boxes']:
                mask_results = self.mask_detector.detect_and_classify_faces(frame, det_results['person_boxes'])

            # Send events to Voice Assistant
            if self.voice_assistant:
                self.voice_assistant.process_detection_events(mask_results, det_results['summary_counts'])

            # Thread-safely update latest inference state
            with self.ai_lock:
                self.latest_detections = det_results
                self.latest_mask_results = mask_results

            time.sleep(0.03)  # Run AI inference ~25-30 times per sec

    def _render_gui_loop(self):
        """Main thread GUI video preview update (runs smoothly at ~30 FPS)."""
        if not self.is_running or self.webcam is None:
            return

        grabbed, frame = self.webcam.read()
        if grabbed and frame is not None:
            # Read latest AI results thread-safely
            with self.ai_lock:
                detections = list(self.latest_detections.get('detections', []))
                person_boxes = list(self.latest_detections.get('person_boxes', []))
                summary_counts = dict(self.latest_detections.get('summary_counts', {}))
                mask_results = list(self.latest_mask_results)

            # Draw bounding boxes on frame
            annotated_frame = self._draw_annotations(frame, detections, mask_results)

            # Convert BGR OpenCV image to PIL Image for CTk
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Resize image to fit canvas container
            canvas_w = max(640, self.video_frame.winfo_width())
            canvas_h = max(360, self.video_frame.winfo_height())
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(canvas_w, canvas_h))

            self.video_label.configure(image=ctk_img, text="")

            # Update Dashboard Stats Labels
            fps = self.webcam.get_fps()
            self.fps_lbl.configure(text=f"FPS: {fps}")
            total_objs = len(detections)
            person_cnt = len(person_boxes)
            self.total_obj_lbl.configure(text=f"Objects Detected: {total_objs}")
            self.persons_lbl.configure(text=f"Persons Detected: {person_cnt}")

            # Update Mask Info List
            self._update_mask_dashboard(mask_results)

        self.after(30, self._render_gui_loop)

    def _draw_annotations(self, frame, detections, mask_results):
        """Draws clear bounding boxes, labels, and mask badges on video frame."""
        annotated = frame.copy()

        # 1. Draw Object Detections (excluding raw person boxes if mask result available)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_name = det['class_name']
            conf = det['confidence']

            if cls_name.lower() == 'person':
                color = (255, 200, 0)  # Cyan/Yellow for Person
            else:
                color = (235, 150, 50)  # Orange for General Objects

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf}%"

            # Draw text backdrop badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Face & Mask Overlay Boxes
        for mask_item in mask_results:
            fx1, fy1, fx2, fy2 = mask_item['box']
            status = mask_item['status']
            conf = mask_item['confidence']

            if status == "MASK ON":
                box_color = (0, 220, 0)  # Bright Green
                badge_text = f"Mask: ON ({conf}%)"
            else:
                box_color = (0, 0, 240)  # Bright Red
                badge_text = f"Mask: OFF ({conf}%)"

            # Face bounding box
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), box_color, 3)

            # Badge background
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (fx1, max(0, fy1 - 24)), (fx1 + tw + 8, max(24, fy1)), box_color, -1)
            cv2.putText(annotated, badge_text, (fx1 + 4, max(17, fy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def _update_mask_dashboard(self, mask_results):
        """Updates the scrollable list of detected persons & mask statuses."""
        for child in self.mask_scroll.winfo_children():
            child.destroy()

        if not mask_results:
            lbl = ctk.CTkLabel(self.mask_scroll, text="No person detected", font=ctk.CTkFont(size=12), text_color="#888888")
            lbl.pack(anchor="w")
            return

        for idx, m in enumerate(mask_results, 1):
            status = m['status']
            conf = m['confidence']
            color = "#28A745" if status == "MASK ON" else "#DC3545"

            entry = ctk.CTkLabel(
                self.mask_scroll,
                text=f"Person {idx} ➔ {status} ({conf}%)",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color
            )
            entry.pack(anchor="w", pady=1)

    # ---------------- Event Handlers ----------------
    def _on_conf_change(self, val):
        pct = int(val * 100)
        self.conf_title.configure(text=f"Detection Confidence: {pct}%")
        if self.object_detector:
            self.object_detector.set_confidence_threshold(val)

    def _on_voice_toggle(self):
        enabled = self.voice_enabled_var.get()
        if self.voice_assistant:
            self.voice_assistant.set_enabled(enabled)

    def _on_speed_change(self, val):
        wpm = int(val)
        self.speed_title.configure(text=f"Voice Speed (WPM): {wpm}")
        if self.voice_assistant:
            self.voice_assistant.set_rate(wpm)

    def _on_volume_change(self, val):
        pct = int(val * 100)
        self.vol_title.configure(text=f"Voice Volume: {pct}%")
        if self.voice_assistant:
            self.voice_assistant.set_volume(val)

    def on_closing(self):
        """Clean shutdown handler."""
        self.stop_camera()
        if self.voice_assistant:
            self.voice_assistant.stop()
        self.destroy()
