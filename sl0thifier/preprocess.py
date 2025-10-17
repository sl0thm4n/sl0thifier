import logging
import os
import subprocess
import sys
import tkinter as tk
import traceback
from datetime import datetime
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageTk

# ─────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR, f"sl0thm4n_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sl0thm4n")

# ─────────────────────────────────────────────────────────
# Paths (고정: 스크립트 기준)
# ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIREFFILE = os.path.join(SCRIPT_DIR, "models", "birefnet.onnx")
REALESRGAN_EXE = os.path.join(SCRIPT_DIR, "realesrgan", "realesrgan-ncnn-vulkan.exe")
REALESRGAN_MODELS = os.path.join(SCRIPT_DIR, "realesrgan", "models")

log.info(f"Python: {sys.executable}")
log.info(f"CWD: {os.getcwd()}")
log.info(f"Script dir: {SCRIPT_DIR}")

# ─────────────────────────────────────────────────────────
# ONNX Runtime (BiRefNet)
# ─────────────────────────────────────────────────────────
try:
    import onnxruntime as ort

    av = ort.get_available_providers()
    log.info(f"ONNX available providers: {av}")
    if "CUDAExecutionProvider" in av:
        PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "DmlExecutionProvider" in av:
        PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        PROVIDERS = ["CPUExecutionProvider"]
except Exception:
    log.error("onnxruntime import 실패:\n" + traceback.format_exc())
    ort = None
    PROVIDERS = ["CPUExecutionProvider"]

bg_session = None
if ort:
    try:
        if os.path.exists(BIREFFILE):
            log.info(f"🧠 Loading BiRefNet: {BIREFFILE}")
            bg_session = ort.InferenceSession(BIREFFILE, providers=PROVIDERS)
            log.info(f"✅ BiRefNet loaded. Providers: {bg_session.get_providers()}")
        else:
            log.warning(f"⚠️ BiRefNet 모델 없음: {BIREFFILE}")
    except Exception:
        log.error("❌ BiRefNet 초기화 실패:\n" + traceback.format_exc())
        bg_session = None


# ─────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────
def ensure_png_alpha(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")


def parse_hex_color(s: str) -> Tuple[int, int, int]:
    s = s.strip().lower()
    if s in ("black", "white"):
        return (0, 0, 0) if s == "black" else (255, 255, 255)
    if s.startswith("#") and len(s) == 7:
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    return (0, 255, 0)  # default green


# ─────────────────────────────────────────────────────────
# RealESRGAN (Vulkan)
# ─────────────────────────────────────────────────────────
def realesrgan_upscale(in_path: str, out_path: str) -> bool:
    if not os.path.exists(REALESRGAN_EXE):
        log.warning(f"⚠️ RealESRGAN 바이너리 없음: {REALESRGAN_EXE}")
        return False
    if not os.path.isdir(REALESRGAN_MODELS):
        log.warning(f"⚠️ RealESRGAN 모델 디렉토리 없음: {REALESRGAN_MODELS}")
        return False

    cmd = [
        REALESRGAN_EXE,
        "-i",
        in_path,
        "-o",
        out_path,
        "-m",
        REALESRGAN_MODELS,
        "-n",
        "realesrgan-x4plus",
        "-s",
        "4",
        "-g",
        "0",
        "-j",
        "1:1:1",
        "-f",
        "png",
    ]
    log.info("▶ RealESRGAN: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=900)
        ok = os.path.exists(out_path)
        log.info(f"✅ Upscale OK? {ok} -> {out_path}")
        return ok
    except subprocess.TimeoutExpired:
        log.error("❌ RealESRGAN timeout")
        return False
    except Exception:
        log.error("❌ RealESRGAN error:\n" + traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────
# BiRefNet background removal
# ─────────────────────────────────────────────────────────
def birefnet_remove_bg(img: Image.Image, bg_choice: str) -> Image.Image:
    log.info("=== BiRefNet background removal START ===")

    if bg_session is None:
        log.warning("BiRefNet session is None. Returning original image.")
        return ensure_png_alpha(img)

    import cv2
    import numpy as np

    try:
        # ---- 입력/출력 이름 자동 탐색 ----
        input_name = bg_session.get_inputs()[0].name
        output_name = bg_session.get_outputs()[0].name
        expected_shape = bg_session.get_inputs()[0].shape  # [1, 3, H, W]
        target_w = int(expected_shape[3])
        target_h = int(expected_shape[2])
        log.info(f"Model expects input size: {target_w}x{target_h}")

        # ---- 이미지 전처리 ----
        np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w = np_img.shape[:2]
        log.debug(f"Input image shape: {np_img.shape}")

        np_in = cv2.resize(np_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        np_in = np.transpose(np_in, (2, 0, 1))[None, :, :, :]
        log.debug(f"Prepared ONNX input shape: {np_in.shape}")

        # ---- 추론 ----
        result = bg_session.run([output_name], {input_name: np_in})[0]
        log.debug(f"ONNX output shape: {result.shape}")

        # ---- 마스크 후처리 ----

        mask = result.squeeze()
        mask = (mask > 0.5).astype("uint8") * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        log.debug(
            f"Mask stats — mean:{mask.mean():.2f}, max:{mask.max()}, min:{mask.min()}"
        )

        rgba = img.convert("RGBA")
        np_rgba = np.array(rgba)

        # ---- 배경 적용 ----
        if bg_choice.lower() == "transparent":
            np_rgba[:, :, 3] = mask
            log.info("Applied transparent background.")
        else:
            r, g, b = parse_hex_color(bg_choice)
            bg = np.full_like(np_rgba, (r, g, b, 255))
            alpha = (mask.astype("float32") / 255.0)[..., None]
            np_rgba = (
                np_rgba.astype("float32") * alpha + bg.astype("float32") * (1 - alpha)
            ).astype("uint8")
            log.info(f"Applied solid background color: {bg_choice}")

        out = Image.fromarray(np_rgba, "RGBA")
        log.info("✅ Background removed successfully (BiRefNet).")
        return out

    except Exception:
        log.error("❌ BiRefNet inference error:\n" + traceback.format_exc())
        return ensure_png_alpha(img)


# ─────────────────────────────────────────────────────────
# Tone
# ─────────────────────────────────────────────────────────
def tone_correction(img: Image.Image) -> Image.Image:
    try:
        enh = ImageEnhance.Color(img)
        img = enh.enhance(1.08)
        enhb = ImageEnhance.Brightness(img)
        img = enhb.enhance(1.02)
        return img
    except Exception:
        log.warning("Tone correction skip:\n" + traceback.format_exc())
        return img


# ─────────────────────────────────────────────────────────────────────────
# Save 512/1024 to {source_dir}\512x512, 1024x1024
# ─────────────────────────────────────────────────────────────────────────
def save_resized_all(img: Image.Image, src_path: str) -> Tuple[str, str]:
    src_dir = os.path.abspath(os.path.dirname(src_path))
    base = os.path.splitext(os.path.basename(src_path))[0]

    out512_dir = os.path.join(src_dir, "512x512")
    out1024_dir = os.path.join(src_dir, "1024x1024")
    os.makedirs(out512_dir, exist_ok=True)
    os.makedirs(out1024_dir, exist_ok=True)

    out512 = os.path.join(out512_dir, f"{base}_512.png")
    out1024 = os.path.join(out1024_dir, f"{base}_1024.png")

    img512 = img.resize((512, 512), Image.Resampling.LANCZOS)
    img1024 = img.resize((1024, 1024), Image.Resampling.LANCZOS)

    img512.save(out512, format="PNG")
    img1024.save(out1024, format="PNG")

    log.info(f"✅ Saved: {out512}")
    log.info(f"✅ Saved: {out1024}")
    return out512, out1024


# ─────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────
THUMB = 128
PAD = 16
GRID_COLS = 5


class FancyUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sl0thm4n Image Preprocessor")
        self.root.geometry("1080x780")
        self.root.configure(bg="#1e1e1e")

        ico_path = os.path.join(SCRIPT_DIR, "assets", "sl0thm4n.ico")
        if os.path.exists(ico_path):
            try:
                self.root.iconbitmap(ico_path)
                log.info(f"Loaded window icon: {ico_path}")
            except Exception as e:
                log.warning(f"Icon load failed: {e}")

        # top bar
        top = tk.Frame(root, bg="#1e1e1e")
        top.pack(fill="x", pady=10)

        self.btn_select = ttk.Button(
            top, text="Select Images", command=self.select_images
        )
        self.btn_select.pack(side="left", padx=8)

        ttk.Label(
            top, text="Background:", foreground="#ddd", background="#1e1e1e"
        ).pack(side="left", padx=6)
        self.bg_var = tk.StringVar(value="transparent")
        self.bg_combo = ttk.Combobox(
            top,
            textvariable=self.bg_var,
            width=14,
            values=["transparent", "black", "white", "#00FF00"],
            state="readonly",
        )
        self.bg_combo.pack(side="left")

        self.tone_var = tk.BooleanVar(value=True)
        self.tone_chk = ttk.Checkbutton(
            top, text="Tone Correction", variable=self.tone_var
        )
        self.tone_chk.pack(side="left", padx=12)

        self.btn_start = ttk.Button(top, text="Start", command=self.start)
        self.btn_start.pack(side="right", padx=8)

        # canvas
        self.canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=10)

        self.items: List[Dict] = []  # {path, thumb_pil, tk_img, cid, x, y}
        self.processing = False

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def select_images(self):
        files = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")],
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
                th = pil.copy()
                th.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(th)
                x = PAD + col * (THUMB + PAD)
                y = PAD + row * (THUMB + PAD)
                cid = self.canvas.create_image(x, y, anchor="nw", image=tkimg)
                self.items.append(
                    {
                        "path": p,
                        "thumb_pil": th,
                        "tk_img": tkimg,
                        "cid": cid,
                        "x": x,
                        "y": y,
                    }
                )
                col += 1
                if col >= GRID_COLS:
                    col = 0
                    row += 1
            except Exception:
                log.warning("thumbnail fail for %s\n%s", p, traceback.format_exc())
        self.canvas.update()

    def start(self):
        if self.processing:
            return
        if not self.items:
            messagebox.showinfo("Info", "Select images first.")
            return
        self.processing = True
        Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        bg_choice = self.bg_var.get()
        tone_on = self.tone_var.get()

        for idx, it in enumerate(self.items):
            src = it["path"]
            log.info(f"\n=== PROCESS {idx + 1}/{len(self.items)} ===")
            log.info(f"SRC: {src}")

            # 1) Upscale
            upscaled_tmp = os.path.join(
                os.path.dirname(src), f"__sloth_upscaled_{os.getpid()}_{idx}.png"
            )
            ok = realesrgan_upscale(src, upscaled_tmp)
            base_path = upscaled_tmp if ok else src
            img = Image.open(base_path).convert("RGBA")

            # 2) Background
            log.info("Calling birefnet_remove_bg() ...")
            img = birefnet_remove_bg(img, bg_choice)
            log.info("birefnet_remove_bg() returned.")

            # 3) Tone
            if tone_on:
                img = tone_correction(img)

            # 4) Save 512 / 1024 (원본 폴더 하위)
            out512, out1024 = save_resized_all(img, src)

            # 5) cross-fade (512 결과)
            self.crossfade_replace(it, out512)

            try:
                if os.path.exists(upscaled_tmp):
                    os.remove(upscaled_tmp)
            except Exception:
                pass

        self.processing = False
        log.info("\n=== ALL DONE ===")
        self.root.after(0, lambda: messagebox.showinfo("Done", "All images processed."))

    def crossfade_replace(
        self, item: Dict, new_path: str, steps: int = 10, delay_ms: int = 24
    ):
        try:
            new_pil = Image.open(new_path).convert("RGBA")
            new_pil.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
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
            log.warning("crossfade error:\n" + traceback.format_exc())


# ─────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FancyUI(root)
        log.info("🚀 GUI launched")
        root.mainloop()
    except Exception:
        log.error("Fatal error:\n" + traceback.format_exc())
    finally:
        input("Press Enter to exit...")
