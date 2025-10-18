from tkinter import Button, Frame, Label, Toplevel

from PIL import Image, ImageTk

try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = getattr(Image, "LANCZOS")


def add_result_button(parent_frame: Frame, result_img: Image.Image) -> None:
    """
    Adds a "Preview Result" button below the given parent frame, which opens a window
    to display the result image when clicked.

    :param parent_frame: The frame below which the button will appear.
    :param result_img: The PIL Image object to be previewed.
    """

    def show_preview():
        preview_window = Toplevel(parent_frame)
        preview_window.title("Result Preview")

        # Resize to fit on screen if too large
        screen_width = preview_window.winfo_screenwidth()
        screen_height = preview_window.winfo_screenheight()
        img_width, img_height = result_img.size

        max_width = screen_width - 100
        max_height = screen_height - 100

        if img_width > max_width or img_height > max_height:
            scale = min(max_width / img_width, max_height / img_height)
            img_width = int(img_width * scale)
            img_height = int(img_height * scale)
            display_img = result_img.resize((img_width, img_height), resample)
        else:
            display_img = result_img

        tk_img = ImageTk.PhotoImage(display_img)

        label = Label(preview_window, image=tk_img)
        label.image = tk_img  # Prevent garbage collection
        label.pack()

    Button(parent_frame, text="👁 Preview Result", command=show_preview).pack(
        pady=(4, 0)
    )
