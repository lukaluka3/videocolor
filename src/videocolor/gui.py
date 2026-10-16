"""Small desktop UI, with all processing outside Tk's main thread."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


def parse_settings(values: dict) -> dict:
    start = float(values["start"])
    seconds = None if values["full"] else float(values["duration"])
    height = None if values.get("no_resize", False) else int(values["height"])
    smoothing = float(values["smoothing"])
    color_strength = float(values.get("color_strength", 0.85))
    tint = float(values.get("tint", 0.0))
    warmth = float(values.get("warmth", 0.0))
    input_size = int(values.get("input_size", 256))
    if not math.isfinite(start) or start < 0:
        raise ValueError("Start time must be a finite number at least zero.")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Preview duration must be a positive finite number.")
    if height is not None and height < 2:
        raise ValueError("Maximum height must be at least 2.")
    if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise ValueError("Smoothing must be between 0 and 1.")
    if not math.isfinite(color_strength) or not 0 <= color_strength <= 1:
        raise ValueError("Color strength must be between 0 and 1.")
    for name, value in (("Tint", tint), ("Warmth", warmth)):
        if not math.isfinite(value) or not -20 <= value <= 20:
            raise ValueError(f"{name} must be a finite number between -20 and 20.")
    if values["model"] not in ("tiny", "large") or values["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Select a supported model and device.")
    if input_size not in (256, 512):
        raise ValueError("Model resolution must be 256 or 512.")
    if not Path(values["source"]).is_file():
        raise ValueError("Choose an existing input video.")
    if not values["output"].strip():
        raise ValueError("Choose a dedicated output job folder.")
    return dict(model=values["model"], device=values["device"], start=start,
                duration=seconds, max_height=height, input_size=input_size,
                smoothing=smoothing, color_strength=color_strength, tint=tint, warmth=warmth,
                resume=bool(values["resume"]))


def load_resume_settings(folder: str | Path) -> tuple[dict, str]:
    """Read job options without trusting stored paths or automatically running it."""
    # The manifest deliberately stores the source name, never its private path.
    manifest = Path(folder) / "videocolor-job.json"
    try:
        job = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(job, dict) or job.get("format") != "videocolor-job" or job.get("version") != 1:
            raise ValueError("Unsupported job format.")
        config = job["config"]
        source_name = job["input"]["name"]
        if not isinstance(source_name, str) or Path(source_name).name != source_name:
            raise ValueError("Invalid source filename.")
        model, device = config["model"], config["device"]
        if model not in ("tiny", "large") or device not in ("auto", "cpu", "cuda"):
            raise ValueError("Unsupported model or device.")
        start = float(config["start"])
        duration = None if config["duration"] is None else float(config["duration"])
        height = config["max_height"]
        smoothing = float(config["smoothing"])
        color_strength = float(config.get("color_strength", 1.0))
        tint, warmth = float(config.get("tint", 0.0)), float(config.get("warmth", 0.0))
        input_size = config["input_size"]
        if not math.isfinite(start) or start < 0:
            raise ValueError("Invalid start time.")
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError("Invalid preview duration.")
        if height is not None and (type(height) is not int or height < 2):
            raise ValueError("Invalid height limit.")
        if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
            raise ValueError("Invalid smoothing strength.")
        if not math.isfinite(color_strength) or not 0 <= color_strength <= 1:
            raise ValueError("Invalid color strength.")
        if any(not math.isfinite(value) or not -20 <= value <= 20 for value in (tint, warmth)):
            raise ValueError("Invalid tint or warmth.")
        if type(input_size) is not int or input_size not in (256, 512):
            raise ValueError("Unsupported model resolution.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot load this VideoColor job's settings: {exc}") from exc
    return {
        "output": str(folder), "resume": True, "model": model, "device": device,
        "start": str(start), "duration": str(duration) if duration is not None else "10",
        "full": duration is None, "height": str(height) if height is not None else "720",
        "no_resize": height is None, "smoothing": str(smoothing), "input_size": str(input_size),
        "color_strength": str(color_strength),
        "tint": str(tint), "warmth": str(warmth),
    }, source_name


def launch() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("The desktop interface requires tkinter. On Ubuntu: sudo apt install python3-tk. You can also use the CLI or Colab notebook.") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("The desktop interface needs a working desktop display and Tcl/Tk installation. Use the CLI or Colab on a headless computer.") from exc
    root.title("VideoColor — black-and-white video colorization")
    root.minsize(760, 680)
    root.columnconfigure(0, weight=1)
    frame = ttk.Frame(root, padding=22)
    frame.grid(sticky="nsew")
    frame.columnconfigure(1, weight=1)
    ttk.Label(frame, text="Give old footage a new perspective", font=("Segoe UI", 19, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
    ttk.Label(frame, text="AI predicts plausible colors. Start with a short preview and review the result.").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(0, 18))

    defaults = dict(source="", output="", model="tiny", device="auto", input_size="256", start="0", duration="10", height="720", smoothing="0.35", color_strength="0.85")
    fields = {name: tk.StringVar(value=value) for name, value in defaults.items()}
    fields.update(tint=tk.StringVar(value="0"), warmth=tk.StringVar(value="0"))
    fields.update(full=tk.BooleanVar(value=False), resume=tk.BooleanVar(value=False), no_resize=tk.BooleanVar(value=False))
    settings_widgets = []
    events: queue.Queue = queue.Queue()
    cancel = threading.Event()
    running = False
    result_path: Path | None = None
    status = tk.StringVar(value="Ready. CPU users: try 1–3 seconds at 240p first.")

    def choose_input() -> None:
        selected = filedialog.askopenfilename(title="Choose a video", filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("All files", "*")])
        if selected:
            fields["source"].set(selected)
            if not fields["output"].get():
                job = Path(selected).parent / "videocolor-outputs" / datetime.now().strftime("preview-%Y%m%d-%H%M%S")
                fields["output"].set(str(job))

    def choose_output() -> None:
        selected = filedialog.askdirectory(title="Choose a parent folder for a NEW job")
        if selected:
            fields["output"].set(str(Path(selected) / datetime.now().strftime("preview-%Y%m%d-%H%M%S")))
            fields["resume"].set(False)

    def choose_resume() -> None:
        selected = filedialog.askdirectory(title="Choose the existing VideoColor job folder")
        if selected:
            try:
                restored, source_name = load_resume_settings(selected)
            except ValueError as exc:
                messagebox.showerror("Cannot load job", str(exc))
                return
            for key, value in restored.items():
                fields[key].set(value)
            if Path(fields["source"].get()).name != source_name:
                fields["source"].set("")
            status.set(f"Saved settings loaded. Select the original input video: {source_name}")

    for row, (key, title, callback) in enumerate((("source", "Input video", choose_input), ("output", "Job folder", choose_output)), 2):
        ttk.Label(frame, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        entry = ttk.Entry(frame, textvariable=fields[key])
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        browse = ttk.Button(frame, text="Browse…", command=callback)
        browse.grid(row=row, column=2, padx=(8, 0))
        settings_widgets.extend(((entry, "normal"), (browse, "normal")))
    controls = ttk.Frame(frame)
    controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=14)
    for column, (key, label, options) in enumerate((("model", "Model", ("tiny", "large")), ("device", "Device", ("auto", "cpu", "cuda")), ("input_size", "Model resolution", ("256", "512")))):
        ttk.Label(controls, text=label).grid(row=0, column=column * 2, padx=(0, 6))
        combo = ttk.Combobox(controls, textvariable=fields[key], values=options, state="readonly", width=10)
        combo.grid(row=0, column=column * 2 + 1, padx=(0, 20))
        settings_widgets.append((combo, "readonly"))
    settings = ttk.Frame(frame)
    settings.grid(row=5, column=0, columnspan=3, sticky="ew")
    for column, (key, label) in enumerate((("start", "Start (seconds)"), ("duration", "Preview (seconds)"), ("height", "Max height (pixels)"), ("smoothing", "Smoothing (0–1)"))):
        ttk.Label(settings, text=label).grid(row=0, column=column, sticky="w", padx=(0, 16))
        entry = ttk.Entry(settings, textvariable=fields[key], width=17)
        entry.grid(row=1, column=column, sticky="w", padx=(0, 16), pady=6)
        settings_widgets.append((entry, "normal"))
    ttk.Label(settings, text="Color strength (0–1)").grid(row=2, column=0, sticky="w", padx=(0, 16))
    color_entry = ttk.Entry(settings, textvariable=fields["color_strength"], width=17)
    color_entry.grid(row=3, column=0, sticky="w", padx=(0, 16), pady=6)
    settings_widgets.append((color_entry, "normal"))
    ttk.Label(settings, text="Lower values reduce saturation. Original colors remain unknown.",
              wraplength=500).grid(row=3, column=1, columnspan=3, sticky="w")
    for column, (key, label) in enumerate((("tint", "Tint (-20 to 20)"), ("warmth", "Warmth (-20 to 20)"))):
        ttk.Label(settings, text=label).grid(row=4, column=column, sticky="w")
        entry = ttk.Entry(settings, textvariable=fields[key], width=17)
        entry.grid(row=5, column=column, sticky="w", pady=6)
        settings_widgets.append((entry, "normal"))
    ttk.Label(settings, text="Tint: green (-) / magenta (+)\nWarmth: blue (-) / yellow (+)",
              wraplength=340).grid(row=4, rowspan=2, column=2, columnspan=2, sticky="w")
    extent = ttk.Frame(frame)
    extent.grid(row=6, column=0, columnspan=3, sticky="w", pady=6)
    for key, label in (("full", "Process to end (may take a long time)"), ("no_resize", "Preserve source size (uses more disk)")):
        checkbox = ttk.Checkbutton(extent, text=label, variable=fields[key])
        checkbox.pack(side="left", padx=(0, 14))
        settings_widgets.append((checkbox, "normal"))
    resume_row = ttk.Frame(frame)
    resume_row.grid(row=7, column=0, columnspan=3, sticky="w")
    resume_checkbox = ttk.Checkbutton(resume_row, text="Resume using identical input and settings", variable=fields["resume"])
    resume_checkbox.pack(side="left")
    resume_button = ttk.Button(resume_row, text="Find existing job…", command=choose_resume)
    resume_button.pack(side="left", padx=12)
    settings_widgets.extend(((resume_checkbox, "normal"), (resume_button, "normal")))
    ttk.Label(frame, text="First use downloads model weights: tiny ≈220 MB, large ≈912 MB. Videos stay on this computer.", wraplength=710).grid(row=8, column=0, columnspan=3, sticky="w", pady=(16, 8))
    bar = ttk.Progressbar(frame, mode="indeterminate")
    bar.grid(row=9, column=0, columnspan=3, sticky="ew", pady=8)
    ttk.Label(frame, textvariable=status, wraplength=710).grid(row=10, column=0, columnspan=3, sticky="w", pady=8)
    buttons = ttk.Frame(frame)
    buttons.grid(row=11, column=0, columnspan=3, sticky="w", pady=12)

    def work(source: str, output: str, options: dict) -> None:
        try:
            from .pipeline import process_video

            def progress(done: int, total: int) -> None:
                if cancel.is_set():
                    raise InterruptedError("Stopped by user")
                events.put(("progress", (done, total)))

            result = process_video(source, output, progress=progress, **options)
            events.put(("complete", result))
        except (KeyboardInterrupt, InterruptedError):
            events.put(("stopped", "Stopped. Completed frames are saved. Resume with identical settings."))
        except Exception as exc:
            events.put(("error", str(exc)))

    def start() -> None:
        nonlocal running, result_path
        if running:
            return
        values = {key: value.get() for key, value in fields.items()}
        try:
            options = parse_settings(values)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check settings", str(exc))
            return
        running, result_path = True, None
        cancel.clear()
        start_button.configure(state="disabled")
        for widget, _ in settings_widgets:
            widget.configure(state="disabled")
        open_button.configure(state="disabled")
        stop_button.configure(state="normal")
        status.set("Reading video and preparing model. First download can take several minutes…")
        bar.configure(mode="indeterminate")
        bar.start()
        threading.Thread(target=work, args=(values["source"], values["output"], options), daemon=True).start()

    def stop() -> None:
        cancel.set()
        status.set("Stop requested. Waiting for the current download, extraction, frame, or encode step to finish…")
        stop_button.configure(state="disabled")

    def open_result() -> None:
        if result_path is None:
            return
        try:
            if os.name == "nt":
                os.startfile(str(result_path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(result_path)])
        except OSError as exc:
            messagebox.showerror("Cannot open player", f"{exc}\nResult: {result_path}")

    def poll() -> None:
        nonlocal running, result_path
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "progress":
                    done, total = payload
                    bar.stop()
                    bar.configure(mode="determinate", maximum=total, value=done)
                    if not cancel.is_set():
                        status.set(f"Frames: {done}/{total}. " + ("Encoding and checking output…" if done == total else "Colorizing…"))
                else:
                    running = False
                    bar.stop()
                    start_button.configure(state="normal")
                    for widget, original_state in settings_widgets:
                        widget.configure(state=original_state)
                    stop_button.configure(state="disabled")
                    if kind == "complete":
                        result_path = payload
                        open_button.configure(state="normal")
                        status.set(f"Saved: {payload}")
                    else:
                        status.set(payload)
                        if kind == "error":
                            messagebox.showerror("Processing could not finish", payload)
        except queue.Empty:
            pass
        root.after(120, poll)

    def close() -> None:
        if running:
            stop()
            messagebox.showinfo("Stopping safely", "A job is still running. Wait for the current step to finish, then close the window.")
        else:
            root.destroy()

    start_button = ttk.Button(buttons, text="Colorize video", command=start)
    start_button.pack(side="left")
    stop_button = ttk.Button(buttons, text="Stop after current step", command=stop, state="disabled")
    stop_button.pack(side="left", padx=10)
    open_button = ttk.Button(buttons, text="Open result", command=open_result, state="disabled")
    open_button.pack(side="left")
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(120, poll)
    root.mainloop()
