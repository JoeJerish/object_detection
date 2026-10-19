import sys
import os

from app_gui import VisionAssistApp


def main():
    print("==================================================")
    print("   Vision Assist AI - Object & Mask Detector     ")
    print("==================================================")

    app = VisionAssistApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
