from tkinter import Tk

from sl0thfier.gui import FancyUI
from sl0thfier.logger import setup_logger

log = setup_logger()

def main():
    try:
        root = Tk()
        app = FancyUI(root)
        log.info("🚀 GUI launched")
        root.mainloop()
    except Exception:
        log.exception("Fatal error")
    finally:
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
