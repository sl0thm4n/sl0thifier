import os
import tkinter as tk
import traceback
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List

from PIL import Image, ImageTk

from sl0thfier.config import load_config
from sl0thfier.imageio import save_resized_all
from sl0thfier.inference import birefnet_remove_bg, realesrgan_upscale
from sl0thfier.logger import setup_logger
from sl0thfier.resize import make_thumbnail

log = setup_logger()
cfg = load_config()

class FancyUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sl0thm4n Image Preprocessor")
        self.root.geometry("1080x780")
        self.root.configure(bg="#1e1e1e")

        ico_path = os.path.join(os.path.dirname(__file__), "assets", "sl0thm4n.ico")
        if os.path.exists(ico_path):
            try:
                self.root.iconbitmap(ico_path)
                log.info("Loaded window icon: %s", ico_path)
            except Exception as e:
                log.warning("Failed to load icon: %s", e)

        # Top bar
        top = tk.Frame(root, bg="#1e1e1e")
        top.pack(fill="x", pady=10)

        self.btn_select = ttk.Button(top, text="Select Images", command=self.select_images)
        self.btn_select.pack(side="left", padx=8)

        ttk.Label(top, text="Background:", foreground="#ddd", background="#1e1e1e").pack(side="left", padx=6)
        self.bg_var = tk.StringVar(value="transparent")
        self.bg_combo = ttk.Combobox(top, textvariable=self.bg_var, width=14,
                                     values=["transparent", "black", "white", "#00FF00"], state="readonly")
        self.bg_combo.pack(side="left")

        self.tone_var = tk.BooleanVar(value=True)
        self.tone_chk = ttk.Checkbutton(top, text="Tone Correction", variable=self.tone_var)
        self.tone_chk.pack(side="left", padx=12)

        self.btn_start = ttk.Button(top, text="Start", command=self.start)
        self.btn_start.pack(side="right", padx=8)

        # Canvas
        self.canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=10)

        self.items: List[Dict] = []
        self.processing = False

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def select_images(self):
        files = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")]
        )
        if not files:
            return
        self.load_thumbnails(list(files))

    def load_thumbnails(self, paths: List[str]):
        self.canvas.delete("all")
        self.items.clear()
        col = row = 0
        for p in paths:
            try:
                pil = Image.open(p).convert("RGBA")
                th = make_thumbnail(pil, cfg.thumb_size)
                tkimg = ImageTk.PhotoImage(th)
                x = cfg.pad + col * (cfg.thumb_size + cfg.pad)
                y = cfg.pad + row * (cfg.thumb_size + cfg.pad)
                cid = self.canvas.create_image(x, y, anchor="nw", image=tkimg)
                self.items.append({"path": p, "thumb_pil": th, "tk_img": tkimg, "cid": cid, "x": x, "y": y})
                col += 1
                if col >= cfg.grid_cols:
                    col = 0
                    row += 1
            except Exception:
                log.warning("Thumbnail failed for %s\n%s", p, traceback.format_exc())
        self.canvas.update()

    def start(self):
        if self.processing:
            return
        if not self.items:
            messagebox.showinfo("Info", "Please select images first.")
            return
        self.processing = True
        Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        bg_choice = self.bg_var.get()
        tone_on = self.tone_var.get()

        for idx, it in enumerate(self.items):
            src = it["path"]
            log.info("\n=== PROCESS %d/%d ===", idx+1, len(self.items))
            log.info("Source: %s", src)

            upscaled_tmp = os.path.join(os.path.dirname(src), f"__sloth_upscaled_{os.getpid()}_{idx}.png")
            ok = realesrgan_upscale(src, upscaled_tmp, cfg.model_paths.realesrgan_exe, cfg.model_paths.realesrgan_models)
            base_path = upscaled_tmp if ok else src
            img = Image.open(base_path).convert("RGBA")

            img = birefnet_remove_bg(img, bg_choice)

            if tone_on:
                from PIL import ImageEnhance
                try:
                    img = ImageEnhance.Color(img).enhance(1.08)
                    img = ImageEnhance.Brightness(img).enhance(1.02)
                except Exception:
                    log.warning("Tone correction failed:\n%s", traceback.format_exc())

            out512, out1024 = save_resized_all(img, src)
            self.crossfade_replace(it, out512)

            try:
                if os.path.exists(upscaled_tmp):
                    os.remove(upscaled_tmp)
            except Exception:
                pass

        self.processing = False
        log.info("\n=== ALL DONE ===")
        self.root.after(0, lambda: messagebox.showinfo("Done", "All images processed."))

    def crossfade_replace(self, item: Dict, new_path: str, steps: int = 10, delay_ms: int = 24):
        try:
            new_pil = Image.open(new_path).convert("RGBA")
            new_pil = make_thumbnail(new_pil, cfg.thumb_size)
            old = item["thumb_pil"].copy().convert("RGBA")

            def step(i=0):
                if i > steps:
                    tkimg = ImageTk.PhotoImage(new_pil)
                    item["tk_img"] = tkimg
                    item["thumb_pil"] = new_pil
                    self.canvas.itemconfigure(item["cid"], image=tkimg)
                    return
                alpha = i / float(steps)
                blended = Image.blend(old, new_pil, alpha)
                tkimg = ImageTk.PhotoImage(blended)
                item["tk_img"] = tkimg
                self.canvas.itemconfigure(item["cid"], image=tkimg)
                self.root.after(delay_ms, lambda: step(i + 1))

            self.root.after(0, step)
        except Exception:
            log.warning("Crossfade error:\n%s", traceback.format_exc())
