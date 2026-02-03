#!/usr/bin/env python3
import os
import queue
import threading
import tkinter as tk
from typing import Optional
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from app import build_summary_lines, process_video, write_output


class PlateApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Australian Dashcam Plate Recognition")
        self.geometry("860x640")
        self.minsize(760, 520)

        self.queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.debug_var = tk.StringVar()

        self.sample_fps_var = tk.StringVar(value="2.0")
        self.max_frames_var = tk.StringVar()
        self.min_conf_var = tk.StringVar(value="0.45")
        self.max_candidates_var = tk.StringVar(value="10")

        self.strict_var = tk.BooleanVar(value=False)
        self.gpu_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        row = 0
        row = self._add_file_row(
            container, row, "Video file", self.video_var, self._browse_video
        )
        row = self._add_file_row(
            container, row, "Output file (optional)", self.output_var, self._browse_output
        )
        row = self._add_file_row(
            container, row, "Debug crop folder", self.debug_var, self._browse_debug
        )

        ttk.Label(container, text="Sample FPS").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(container, textvariable=self.sample_fps_var, width=12).grid(
            row=row, column=1, sticky="w", pady=4
        )
        row += 1

        ttk.Label(container, text="Max frames (optional)").grid(
            row=row, column=0, sticky="w", pady=4
        )
        ttk.Entry(container, textvariable=self.max_frames_var, width=12).grid(
            row=row, column=1, sticky="w", pady=4
        )
        row += 1

        ttk.Label(container, text="Min confidence").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(container, textvariable=self.min_conf_var, width=12).grid(
            row=row, column=1, sticky="w", pady=4
        )
        row += 1

        ttk.Label(container, text="Max candidates per frame").grid(
            row=row, column=0, sticky="w", pady=4
        )
        ttk.Entry(container, textvariable=self.max_candidates_var, width=12).grid(
            row=row, column=1, sticky="w", pady=4
        )
        row += 1

        options_frame = ttk.Frame(container)
        options_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Checkbutton(
            options_frame, text="Strict AU patterns", variable=self.strict_var
        ).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(options_frame, text="Use GPU OCR", variable=self.gpu_var).grid(
            row=0, column=1, sticky="w"
        )
        row += 1

        self.run_button = ttk.Button(container, text="Run scan", command=self.start_processing)
        self.run_button.grid(row=row, column=0, sticky="w", pady=(6, 10))

        ttk.Label(container, textvariable=self.status_var).grid(
            row=row, column=1, sticky="w", pady=(6, 10)
        )
        row += 1

        self.output_text = ScrolledText(
            container, height=16, wrap=tk.WORD, state="disabled"
        )
        self.output_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        container.rowconfigure(row, weight=1)

    def _add_file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=(0, 6), pady=4
        )
        ttk.Button(parent, text="Browse", command=browse_command).grid(
            row=row, column=2, sticky="w", pady=4
        )
        return row + 1

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select dashcam video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save output file",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _browse_debug(self) -> None:
        path = filedialog.askdirectory(title="Select debug output folder")
        if path:
            self.debug_var.set(path)

    def append_log(self, message: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")

    def start_processing(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Processing", "A scan is already running.")
            return

        video_path = self.video_var.get().strip()
        if not video_path:
            messagebox.showerror("Missing video", "Please select a dashcam video.")
            return
        if not os.path.exists(video_path):
            messagebox.showerror("Missing video", "The selected video does not exist.")
            return

        try:
            sample_fps = float(self.sample_fps_var.get().strip() or "0")
            min_conf = float(self.min_conf_var.get().strip() or "0")
            max_candidates = int(self.max_candidates_var.get().strip() or "0")
            max_frames_raw = self.max_frames_var.get().strip()
            max_frames = int(max_frames_raw) if max_frames_raw else None
        except ValueError:
            messagebox.showerror("Invalid input", "Please check numeric fields.")
            return

        debug_dir = self.debug_var.get().strip() or None
        output_path = self.output_var.get().strip() or None

        self.run_button.configure(state="disabled")
        self.status_var.set("Processing...")
        self.append_log("Starting scan...")

        args = (
            video_path,
            sample_fps,
            max_frames,
            min_conf,
            max_candidates,
            self.strict_var.get(),
            self.gpu_var.get(),
            debug_dir,
            output_path,
        )
        self.worker_thread = threading.Thread(target=self._worker, args=args, daemon=True)
        self.worker_thread.start()
        self.after(200, self._poll_queue)

    def _worker(
        self,
        video_path: str,
        sample_fps: float,
        max_frames: Optional[int],
        min_conf: float,
        max_candidates: int,
        strict_patterns: bool,
        use_gpu: bool,
        debug_dir: Optional[str],
        output_path: Optional[str],
    ) -> None:
        try:
            data = process_video(
                video_path=video_path,
                sample_fps=sample_fps,
                max_frames=max_frames,
                min_confidence=min_conf,
                max_candidates=max_candidates,
                strict_patterns=strict_patterns,
                use_gpu=use_gpu,
                debug_dir=debug_dir,
            )
            if output_path:
                write_output(output_path, data)
                self.queue.put(("log", f"Wrote results to {output_path}"))
            summary = "\n".join(build_summary_lines(data))
            summary += (
                f"\n\nFrames processed: {data['frames_processed']}\n"
                f"Processing time: {data['processing_seconds']}s\n"
                f"Plates found: {data['plates_found']}\n"
            )
            self.queue.put(("done", summary))
        except SystemExit as exc:
            self.queue.put(("error", str(exc)))
        except Exception as exc:  # pragma: no cover - UI display
            self.queue.put(("error", f"Unexpected error: {exc}"))

    def _poll_queue(self) -> None:
        try:
            while True:
                message_type, payload = self.queue.get_nowait()
                if message_type == "log":
                    self.append_log(payload)
                elif message_type == "done":
                    self.append_log(payload)
                    self.status_var.set("Finished.")
                elif message_type == "error":
                    self.append_log(f"Error: {payload}")
                    messagebox.showerror("Processing error", payload)
                    self.status_var.set("Error.")
        except queue.Empty:
            pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.after(200, self._poll_queue)
        else:
            self.run_button.configure(state="normal")


def main() -> None:
    app = PlateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
