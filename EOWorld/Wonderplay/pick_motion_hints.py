#!/usr/bin/env python3
"""Pick motion hint line coordinates from an image.

Drag with the left mouse button to create a line segment. Each segment is
reported as [x1, y1, x2, y2] in image coordinates, with the origin at the
top-left corner, x increasing to the right, and y increasing downward.
"""

from __future__ import annotations

import argparse
import pprint
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw several line segments on an image and print coordinates."
    )
    parser.add_argument("image", type=Path, help="Path to the image to annotate.")
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Do not scale the image to fit the screen.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the final coordinate list to the clipboard on exit.",
    )
    return parser.parse_args()


class LinePicker:
    def __init__(self, image_path: Path, fit_to_screen: bool = True, copy_to_clipboard: bool = False):
        self.image_path = image_path
        self.copy_to_clipboard = copy_to_clipboard
        self.lines: list[list[int]] = []
        self.canvas_items: list[list[int]] = []
        self.temp_line_id: int | None = None
        self.start_canvas: tuple[int, int] | None = None
        self.start_image: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title(f"Line picker - {image_path.name}")
        self.root.configure(bg="#202124")
        self.root.protocol("WM_DELETE_WINDOW", self.finish)

        image = Image.open(image_path).convert("RGB")
        self.original_size = image.size

        if fit_to_screen:
            screen_w = max(1, self.root.winfo_screenwidth() - 80)
            screen_h = max(1, self.root.winfo_screenheight() - 120)
            scale = min(
                1.0,
                screen_w / self.original_size[0],
                screen_h / self.original_size[1],
            )
        else:
            scale = 1.0

        self.display_size = (
            max(1, int(round(self.original_size[0] * scale))),
            max(1, int(round(self.original_size[1] * scale))),
        )
        self.scale_x = self.original_size[0] / self.display_size[0]
        self.scale_y = self.original_size[1] / self.display_size[1]

        if self.display_size != self.original_size:
            image = image.resize(self.display_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_size[0],
            height=self.display_size[1],
            highlightthickness=0,
            cursor="crosshair",
            bg="black",
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.undo_last)
        self.root.bind("<BackSpace>", self.undo_last)
        self.root.bind("<Delete>", self.undo_last)
        self.root.bind("c", self.clear_all)
        self.root.bind("C", self.clear_all)
        self.root.bind("<Return>", self.finish)
        self.root.bind("<Escape>", self.finish)

    def canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        ix = int(round(x * self.scale_x))
        iy = int(round(y * self.scale_y))
        ix = max(0, min(self.original_size[0] - 1, ix))
        iy = max(0, min(self.original_size[1] - 1, iy))
        return ix, iy

    def on_press(self, event: tk.Event) -> None:
        self.start_canvas = (event.x, event.y)
        self.start_image = self.canvas_to_image(event.x, event.y)
        if self.temp_line_id is not None:
            self.canvas.delete(self.temp_line_id)
        self.temp_line_id = self.canvas.create_line(
            event.x,
            event.y,
            event.x,
            event.y,
            fill="#ff5c5c",
            width=3,
        )

    def on_drag(self, event: tk.Event) -> None:
        if self.temp_line_id is None or self.start_canvas is None:
            return
        self.canvas.coords(
            self.temp_line_id,
            self.start_canvas[0],
            self.start_canvas[1],
            event.x,
            event.y,
        )

    def on_release(self, event: tk.Event) -> None:
        if self.start_canvas is None or self.start_image is None:
            return

        end_image = self.canvas_to_image(event.x, event.y)
        if end_image == self.start_image:
            self.cleanup_temp()
            return

        coords = [self.start_image[0], self.start_image[1], end_image[0], end_image[1]]
        self.lines.append(coords)

        if self.temp_line_id is not None:
            self.canvas.delete(self.temp_line_id)
            self.temp_line_id = None

        line_id = self.canvas.create_line(
            self.start_canvas[0],
            self.start_canvas[1],
            event.x,
            event.y,
            fill="#00e676",
            width=3,
        )
        self.canvas_items.append([line_id])
        print(coords, flush=True)

        self.start_canvas = None
        self.start_image = None

    def cleanup_temp(self) -> None:
        if self.temp_line_id is not None:
            self.canvas.delete(self.temp_line_id)
            self.temp_line_id = None
        self.start_canvas = None
        self.start_image = None

    def undo_last(self, event: tk.Event | None = None) -> None:
        self.cleanup_temp()
        if not self.lines:
            return
        self.lines.pop()
        item_ids = self.canvas_items.pop()
        for item_id in item_ids:
            self.canvas.delete(item_id)
        print(f"undo -> {len(self.lines)} segments", flush=True)

    def clear_all(self, event: tk.Event | None = None) -> None:
        self.cleanup_temp()
        for item_ids in self.canvas_items:
            for item_id in item_ids:
                self.canvas.delete(item_id)
        self.lines.clear()
        self.canvas_items.clear()
        print("cleared", flush=True)

    def finish(self, event: tk.Event | None = None) -> None:
        result = pprint.pformat(self.lines, width=88)
        print("\nfinal coordinates:")
        print(result)
        if self.copy_to_clipboard:
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(result)
                self.root.update_idletasks()
                print("(copied to clipboard)")
            except tk.TclError as exc:
                print(f"(clipboard unavailable: {exc})", file=sys.stderr)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    picker = LinePicker(
        args.image,
        fit_to_screen=not args.no_fit,
        copy_to_clipboard=args.copy,
    )
    picker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
