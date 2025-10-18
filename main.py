import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image
from tkinterdnd2 import DND_FILES, TkinterDnD

from sl0thifier.models import KingSl0th


class Sl0thifyGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Sl0thify")
        self.root.geometry("600x800")
        self.root.resizable(False, False)

        self.remove_bg_var = tk.BooleanVar(value=False)
        self.bg_color_var = tk.StringVar(value="None")

        # Top options
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=(10, 0))

        tk.Checkbutton(
            options_frame, text="Remove Background", variable=self.remove_bg_var
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(options_frame, text="New Background Color:").pack(side=tk.LEFT)
        tk.OptionMenu(
            options_frame, self.bg_color_var, "None", "White", "Black", "Green"
        ).pack(side=tk.LEFT, padx=5)

        # Drop zone
        self.drop_label = tk.Label(
            self.root,
            text="Drop files or folders here",
            relief="ridge",
            borderwidth=2,
            width=64,
            height=16,
        )
        self.drop_label.pack(pady=10)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", length=512, mode="determinate"
        )
        self.progress.pack(pady=(0, 20))

        self.king = KingSl0th(model_name="realesrgan-x4plus", width=512, height=512)

        self.root.mainloop()

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        all_files = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    all_files += [
                        os.path.join(root, f)
                        for f in files
                        if f.lower().endswith(("png", "jpg", "jpeg"))
                    ]
            else:
                if path.lower().endswith(("png", "jpg", "jpeg")):
                    all_files.append(path)

        self.process_files(all_files)

    def process_files(self, files):
        total = len(files)
        for i, file_path in enumerate(files):
            self.progress["value"] = (i / total) * 100
            self.root.update_idletasks()
            try:
                with Image.open(file_path) as img:
                    output_base_dir = Path(file_path).parent / "sl0thified"
                    output_base_dir.mkdir(parents=True, exist_ok=True)

                    result_img = self.king.sl0thify(
                        img,
                        output_path=output_base_dir,
                        output_width=img.width,
                        output_height=img.height,
                        remove_bg=self.remove_bg_var.get(),
                        bg_color=self.bg_color_var.get(),
                    )

                    output_path = (
                        output_base_dir
                        / f"{Path(file_path).stem}-sl0thified-{img.width}x{img.height}{Path(file_path).suffix}"
                    )
                    print(f"✅ Processed {file_path} -> {output_path}")
                    result_img.save(output_path)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        self.progress["value"] = 100


if __name__ == "__main__":
    Sl0thifyGUI()
