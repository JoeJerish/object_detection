import cv2
import time
import threading
import numpy as np


class WebcamStream:
    """
    Threaded webcam capture stream to ensure zero camera lag.
    Captures frames asynchronously in a dedicated thread.
    """

    def __init__(self, src=0, target_width=1280, target_height=720, target_fps=30):
        self.src = src
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps

        self.cap = None
        self.frame = None
        self.grabbed = False
        self.stopped = True
        self.lock = threading.Lock()

        # FPS metrics
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.actual_width = 640
        self.actual_height = 480

        # Contrast / Sharpness enhancement settings
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhance_lighting = True

    def initialize_camera(self):
        """Probes and sets up the highest practical supported resolution."""
        # Try DirectShow backend on Windows first for fast startup
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video camera source index {self.src}")

        # Priority resolution testing list
        resolutions = [
            (1920, 1080),
            (1280, 720),
            (960, 540),
            (640, 480)
        ]

        # Filter resolutions up to requested target
        res_candidates = [r for r in resolutions if r[0] <= self.target_width or r == (640, 480)]
        if not res_candidates:
            res_candidates = resolutions

        set_success = False
        for w, h in res_candidates:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_w > 0 and actual_h > 0:
                self.actual_width = actual_w
                self.actual_height = actual_h
                set_success = True
                print(f"[Camera] Configured resolution: {self.actual_width}x{self.actual_height}")
                break

        if not set_success:
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Enable auto-focus and auto-exposure if supported
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Read initial frame
        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed or self.frame is None:
            raise RuntimeError("Failed to read initial frame from camera.")

    def start(self):
        """Starts background frame reader thread."""
        self.initialize_camera()
        self.stopped = False
        self.start_time = time.time()
        self.frame_count = 0

        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Continuously grab frames in background thread."""
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                break

            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Frame light enhancement if needed
            if self.enhance_lighting:
                frame = self._enhance_frame(frame)

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
                self.frame_count += 1

                # Calculate live FPS
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

            time.sleep(0.005)

    def _enhance_frame(self, frame):
        """Applies fast lighting adjustment to improve visual quality."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception:
            return frame

    def read(self):
        """Returns the latest grabbed frame thread-safely."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def get_fps(self):
        with self.lock:
            return round(self.fps, 1)

    def get_resolution(self):
        return self.actual_width, self.actual_height

    def stop(self):
        """Stops thread and releases webcam hardware."""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[Camera] Stopped webcam stream.")
