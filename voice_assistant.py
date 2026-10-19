import pyttsx3
import threading
import queue
import time


class VoiceAssistant:
    """
    Non-blocking Windows SAPI5 Text-To-Speech assistant.
    Runs speech in a dedicated daemon thread.
    Includes smart cooldowns to eliminate repetitive voice alerts.
    """

    def __init__(self, rate=165, volume=1.0, cooldown_seconds=8.0):
        self.speech_queue = queue.PriorityQueue()
        self.is_enabled = True
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken_times = {}

        self.requested_rate = rate
        self.requested_volume = volume
        self.stopped = False

        # Current state tracker to prevent duplicate announcements
        self.last_mask_summary = None

        # Start background TTS thread
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    def _speech_worker(self):
        """Worker thread loop initializing COM and processing speech queue."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        try:
            engine = pyttsx3.init('sapi5')
        except Exception:
            engine = pyttsx3.init()

        # Search for female English voice
        voices = engine.getProperty('voices')
        female_voice_id = None
        selected_voice_name = "Default"

        for v in voices:
            name_lower = v.name.lower()
            if 'zira' in name_lower or 'eva' in name_lower or 'hazel' in name_lower or 'female' in name_lower:
                female_voice_id = v.id
                selected_voice_name = v.name
                break

        if female_voice_id:
            engine.setProperty('voice', female_voice_id)
            print(f"[VoiceAssistant] Female voice selected: {selected_voice_name}")
        else:
            print("[VoiceAssistant] Female voice not found. Using default voice.")
            print("[VoiceAssistant] To install Windows Female voice: Go to Windows Settings -> Time & Language -> Speech -> Add voices -> Select 'English (US) - Zira' or 'English (UK) - Hazel'.")

        engine.setProperty('rate', self.requested_rate)
        engine.setProperty('volume', self.requested_volume)

        # Worker loop
        while not self.stopped:
            try:
                # Retrieve next item from priority queue (priority, text)
                priority, text = self.speech_queue.get(timeout=0.2)
                if text and self.is_enabled:
                    # Dynamically update rate and volume
                    engine.setProperty('rate', self.requested_rate)
                    engine.setProperty('volume', self.requested_volume)

                    engine.say(text)
                    engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceAssistant] Speech error: {e}")

    def speak(self, text, priority=2, force=False):
        """
        Enqueues speech item.
        Priority: 1 (High/Mask Warning), 2 (Normal Object Alert), 3 (Low/Info).
        """
        if not self.is_enabled and not force:
            return

        now = time.time()
        event_key = text.strip().lower()

        # Cooldown check unless forced or high priority safety alert
        if not force and priority > 1:
            last_time = self.last_spoken_times.get(event_key, 0)
            if now - last_time < self.cooldown_seconds:
                return

        self.last_spoken_times[event_key] = now

        # Prevent queue overload
        if self.speech_queue.qsize() > 5:
            # Clear older low-priority messages
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                pass

        self.speech_queue.put((priority, text))

    def process_detection_events(self, mask_results, summary_counts):
        """
        Evaluates detection frames and speaks intelligent, non-repetitive events.
        """
        if not self.is_enabled:
            return

        now = time.time()

        # 1. Mask state alerts (Highest priority)
        if mask_results:
            mask_off_count = sum(1 for m in mask_results if m['status'] == 'MASK OFF')
            mask_on_count = sum(1 for m in mask_results if m['status'] == 'MASK ON')

            current_mask_summary = (mask_off_count, mask_on_count)
            if current_mask_summary != self.last_mask_summary:
                self.last_mask_summary = current_mask_summary

                if mask_off_count > 0:
                    if mask_off_count == 1:
                        msg = "Person detected without a mask."
                    else:
                        msg = f"{mask_off_count} persons detected without masks."
                    self.speak(msg, priority=1, force=True)

                elif mask_on_count > 0:
                    if mask_on_count == 1:
                        msg = "Person detected wearing a mask."
                    else:
                        msg = f"{mask_on_count} persons detected wearing masks."
                    self.speak(msg, priority=2)

        # 2. General Object Detection Alerts
        important_objects = ['Car', 'Bicycle', 'Mobile phone', 'Laptop', 'Bottle', 'Chair', 'Backpack', 'Dog', 'Cat']
        for obj_name, count in summary_counts.items():
            if obj_name in important_objects:
                last_time = self.last_spoken_times.get(obj_name.lower(), 0)
                if now - last_time > 12.0:  # 12 sec cooldown for general objects
                    if count == 1:
                        msg = f"{obj_name} detected."
                    else:
                        msg = f"{count} {obj_name}s detected."
                    self.speak(msg, priority=3)

    def set_enabled(self, enabled: bool):
        self.is_enabled = enabled

    def set_rate(self, rate: int):
        self.requested_rate = max(80, min(300, int(rate)))

    def set_volume(self, volume: float):
        self.requested_volume = max(0.0, min(1.0, float(volume)))

    def stop(self):
        self.stopped = True
