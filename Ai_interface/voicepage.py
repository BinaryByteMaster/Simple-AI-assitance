import tkinter as tk
from tkinter import ttk, messagebox
import math
import queue
import json
import sys
import os
import sounddevice as sd
_ai_core_path = os.path.join(os.path.dirname(__file__), "..", "Ai core")
sys.path.append(os.path.abspath(_ai_core_path))
import threading
import pyaudio
from deepgramAi import (
    start_agent_connection,
    send_audio_chunk,
    stop_agent_connection,
)
# ---- option lists for the settings form -----------------------------
AUDIO_ENCODINGS = ["linear16", "mulaw", "alaw", "opus", "flac"]
SAMPLE_RATES = [8000, 16000, 24000, 44100, 48000]
OUTPUT_CONTAINERS = ["none", "wav"]

LISTEN_PROVIDER_TYPES = ["deepgram"]
LISTEN_VERSIONS = ["v1", "v2"]
LISTEN_MODELS = ["flux-general-en", "nova-2", "nova-2-general", "nova-2-conversationalai", "enhanced", "base"]

SPEAK_PROVIDER_TYPES = ["deepgram"]
SPEAK_MODELS = [
    "aura-asteria-en", "aura-luna-en", "aura-stella-en", "aura-athena-en",
    "aura-hera-en", "aura-orion-en", "aura-arcas-en", "aura-perseus-en",
    "aura-angus-en", "aura-orpheus-en", "aura-helios-en", "aura-zeus-en",
]

THINK_PROVIDER_TYPES = ["anthropic", "open_ai", "groq"]
THINK_MODELS = [
    "claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001",
    "gpt-4o", "gpt-4o-mini",
]

# Colour stops for the ring gradient, going around the full circle.
GRADIENT_STOPS = [
    (0.00, (10, 10, 14)),
    (0.10, (30, 200, 120)),   # green
    (0.28, (60, 210, 210)),   # cyan
    (0.45, (90, 130, 235)),   # blue
    (0.60, (160, 90, 225)),   # purple
    (0.75, (215, 80, 200)),   # magenta
    (0.88, (235, 130, 175)),  # pink
    (1.00, (10, 10, 14)),
]


def _lerp(a, b, t):
    return a + (b - a) * t


def _gradient_color(t):
    t = t % 1.0
    for i in range(len(GRADIENT_STOPS) - 1):
        p0, c0 = GRADIENT_STOPS[i]
        p1, c1 = GRADIENT_STOPS[i + 1]
        if p0 <= t <= p1:
            local_t = 0 if p1 == p0 else (t - p0) / (p1 - p0)
            r = int(_lerp(c0[0], c1[0], local_t))
            g = int(_lerp(c0[1], c1[1], local_t))
            b = int(_lerp(c0[2], c1[2], local_t))
            return "#{:02x}{:02x}{:02x}".format(r, g, b)
    return "#000000"


def round_rectangle(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class DeepgramVoiceAssistantGUI(tk.Tk):

    STATE_SPEED = {"idle": 0.6, "listening": 2.2, "processing": 3.2, "speaking": 1.6}
    STATE_PULSE = {"idle": 0.0, "listening": 10.0, "processing": 4.0, "speaking": 14.0}

    def __init__(self):
        super().__init__()
        self.title("Deepgram Voice Assistant")
        self.geometry("1040x680")
        self.minsize(860, 580)
        self.configure(bg="#000000")
        self.connection = None
        self.connection_cm = None
        self.mic_thread = None
        self.is_running = False

       
        

        self.is_active = False
        self.state = "idle"
        self._phase = 0.0
        self._pulse_t = 0.0
        self.last_message = ("assistant", "Hello?")
        self._end_button_bbox = None

        # ---- settings state, mirrors the Deepgram Agent Settings JSON ---
        self.input_encoding = tk.StringVar(value="linear16")
        self.input_sample_rate = tk.IntVar(value=24000)

        self.output_encoding = tk.StringVar(value="linear16")
        self.output_sample_rate = tk.IntVar(value=24000)
        self.output_container = tk.StringVar(value="none")

        self.listen_provider_type = tk.StringVar(value="deepgram")
        self.listen_version = tk.StringVar(value="v2")
        self.listen_model = tk.StringVar(value="flux-general-en")

        self.speak_provider_type = tk.StringVar(value="deepgram")
        self.speak_model = tk.StringVar(value="aura-asteria-en")

        self.think_provider_type = tk.StringVar(value="anthropic")
        self.think_model = tk.StringVar(value="claude-sonnet-5")
        # think prompt lives in a Text widget (multiline), created in _build_settings_panel

        self.greeting = tk.StringVar(value="Hello! How may I help you?")

        self._msg_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self._build_style()
        self._build_layout()
        self._animate()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        panel, fg, accent = "#25262f", "#f2f2f2", "#13ef95"

        style.configure("TFrame", background=panel)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Panel.TLabel", background=panel, foreground=fg, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=panel, foreground="#9a9ea8", font=("Segoe UI", 8))
        style.configure("Header.TLabel", background=panel, foreground=fg, font=("Segoe UI", 12, "bold"))
        style.configure("Section.TLabel", background=panel, foreground=accent, font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=panel, foreground=fg, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", panel)])
        style.configure("TCombobox", fieldbackground="white")
        style.configure("TSpinbox", fieldbackground="white")
        style.configure(
            "Accent.TButton", background=accent, foreground="#0b0c10",
            font=("Segoe UI", 11, "bold"), padding=8,
        )
        style.map("Accent.TButton", background=[("active", "#0fd486")])

        style.configure("TNotebook", background="#000000", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1a1b22", foreground=fg, padding=(14, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", panel)], foreground=[("selected", fg)])

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # LEFT — orb canvas
        left = tk.Frame(self, bg="#000000")
        left.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(left, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)

        # RIGHT — tabbed Settings / Transcript
        right = ttk.Frame(self, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")

        settings_tab = ttk.Frame(notebook, style="Panel.TFrame")
        transcript_tab = ttk.Frame(notebook, style="Panel.TFrame")
        notebook.add(settings_tab, text="Settings")
        notebook.add(transcript_tab, text="Transcript")

        self._build_settings_tab(settings_tab)
        self._build_transcript_panel(transcript_tab)
    # edit the AI Agent Settings JSON file with the current form values
    def edit_settings_json(self):
        json_path = os.path.join(os.path.dirname(__file__), "..", "Ai.json")

        # read existing settings first, so anything not covered here is preserved
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data["type"] = "Settings"

        # ---- audio ----------------------------------------------------------
        data.setdefault("audio", {})
        data["audio"]["input"] = {
            "encoding": self.input_encoding.get(),
            "sample_rate": self.input_sample_rate.get(),
        }
        data["audio"]["output"] = {
            "encoding": self.output_encoding.get(),
            "sample_rate": self.output_sample_rate.get(),
            "container": self.output_container.get(),
        }

        # ---- agent ------------------------------------------------------------
        data.setdefault("agent", {})

        data["agent"]["listen"] = {
            "provider": {
                "type": self.listen_provider_type.get(),
                "model": self.listen_model.get(),
                "version": self.listen_version.get(),
            }
        }

        data["agent"]["think"] = {
            "provider": {
                "type": self.think_provider_type.get(),
                "model": self.think_model.get(),
            },
            # pulled from the multiline Text widget — see note below
            "prompt": self.think_prompt_text.get("1.0", "end").strip(),
        }

        data["agent"]["speak"] = {
            "provider": {
                "type": self.speak_provider_type.get(),
                "model": self.speak_model.get(),
            }
        }

        data["agent"]["greeting"] = self.greeting.get()

        # ---- write back -------------------------------------------------------
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            messagebox.showerror("Save Failed", f"Couldn't save settings:\n{e}")
            return

        messagebox.showinfo("Settings Saved", "AI Agent settings updated ✓")

    # ------------------------------------------------------------------
    # Settings tab — scrollable, mirrors the Agent Settings JSON
    # ------------------------------------------------------------------
    def _build_settings_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, bg="#25262f", highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas, style="Panel.TFrame", padding=16)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(inner_id, width=e.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._populate_settings_form(inner)

    def _section_label(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(14, 4))

    def _field_label(self, parent, text):
        ttk.Label(parent, text=text, style="Panel.TLabel").pack(anchor="w")

    def _populate_settings_form(self, parent):
        ttk.Label(parent, text="Deepgram Agent Settings", style="Header.TLabel").pack(anchor="w")

        # ---- audio.input ----------------------------------------------
        self._section_label(parent, "Audio — Input")
        self._field_label(parent, "Encoding")
        ttk.Combobox(parent, textvariable=self.input_encoding, values=AUDIO_ENCODINGS, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Sample rate (Hz)")
        ttk.Combobox(parent, textvariable=self.input_sample_rate, values=SAMPLE_RATES, state="readonly").pack(
            fill="x", pady=(2, 4)
        )

        # ---- audio.output ----------------------------------------------
        self._section_label(parent, "Audio — Output")
        self._field_label(parent, "Encoding")
        ttk.Combobox(parent, textvariable=self.output_encoding, values=AUDIO_ENCODINGS, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Sample rate (Hz)")
        ttk.Combobox(parent, textvariable=self.output_sample_rate, values=SAMPLE_RATES, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Container")
        ttk.Combobox(parent, textvariable=self.output_container, values=OUTPUT_CONTAINERS, state="readonly").pack(
            fill="x", pady=(2, 4)
        )

        # ---- agent.listen ----------------------------------------------
        self._section_label(parent, "Agent — Listen")
        self._field_label(parent, "Provider type")
        ttk.Combobox(parent, textvariable=self.listen_provider_type, values=LISTEN_PROVIDER_TYPES, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Version")
        ttk.Combobox(parent, textvariable=self.listen_version, values=LISTEN_VERSIONS, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Model")
        ttk.Combobox(parent, textvariable=self.listen_model, values=LISTEN_MODELS).pack(fill="x", pady=(2, 4))

        # ---- agent.speak ----------------------------------------------
        self._section_label(parent, "Agent — Speak")
        self._field_label(parent, "Provider type")
        ttk.Combobox(parent, textvariable=self.speak_provider_type, values=SPEAK_PROVIDER_TYPES, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Model")
        ttk.Combobox(parent, textvariable=self.speak_model, values=SPEAK_MODELS).pack(fill="x", pady=(2, 4))

        # ---- agent.think ----------------------------------------------
        self._section_label(parent, "Agent — Think")
        self._field_label(parent, "Provider type")
        ttk.Combobox(parent, textvariable=self.think_provider_type, values=THINK_PROVIDER_TYPES, state="readonly").pack(
            fill="x", pady=(2, 8)
        )
        self._field_label(parent, "Model")
        ttk.Combobox(parent, textvariable=self.think_model, values=THINK_MODELS).pack(fill="x", pady=(2, 8))
        self._field_label(parent, "Prompt")
        self.think_prompt_text = tk.Text(
            parent, height=4, wrap="word", bg="#101116", fg="#e8e8e8",
            insertbackground="white", font=("Consolas", 9), relief="flat", padx=8, pady=6,
        )
        self.think_prompt_text.pack(fill="x", pady=(2, 4))

        # ---- agent.greeting ---------------------------------------------
        self._section_label(parent, "Agent — Greeting")
        ttk.Entry(parent, textvariable=self.greeting).pack(fill="x", pady=(2, 4))

        # ---- JSON preview / export ---------------------------------------
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(16, 10))
        ttk.Button(parent, text="Edit AI Agent Settings", command=self.edit_settings_json).pack(fill="x")

        status_row = ttk.Frame(parent, style="Panel.TFrame")
        status_row.pack(fill="x", pady=(10, 0))
        self.status_note = ttk.Label(status_row, text="Status: Idle", style="Muted.TLabel")
        self.status_note.pack(side="left")

    

    # ------------------------------------------------------------------
    # Transcript panel
    # ------------------------------------------------------------------
    def _build_transcript_panel(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        header_row = ttk.Frame(parent, style="Panel.TFrame", padding=(16, 16, 16, 6))
        header_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(header_row, text="Transcript", style="Header.TLabel").pack(side="left")
        ttk.Button(header_row, text="Clear", command=self._clear_transcript).pack(side="right")

        text_frame = ttk.Frame(parent, style="Panel.TFrame", padding=(16, 0, 16, 16))
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self.transcript = tk.Text(
            text_frame, wrap="word", bg="#101116", fg="#e8e8e8", insertbackground="white",
            font=("Consolas", 10), relief="flat", padx=10, pady=8,
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=scrollbar.set)
        self.transcript.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.transcript.configure(state="disabled")

        self.transcript.tag_config("user", foreground="#6cc4ff")
        self.transcript.tag_config("assistant", foreground="#13ef95")
        self.transcript.tag_config("system", foreground="#8a8f98")

        self._append_transcript("system", "Ready. Configure settings and press 'Start Conversation'.")

    def _append_transcript(self, who, text):
        self.transcript.configure(state="normal")
        prefix = {"user": "You: ", "assistant": "Assistant: ", "system": "* "}[who]
        self.transcript.insert("end", prefix + text + "\n", who)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")
    def _append_log(self, role, content):
        """Display system/agent log messages in the log area."""
        try:
            self._append_transcript(role, content)
        except Exception as e:
            print(f"Log display error: {e}")
    def _clear_transcript(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._append_transcript("system", "Transcript cleared.")

    # ------------------------------------------------------------------
    # Orb drawing
    # ------------------------------------------------------------------
    def _draw(self):
        c = self.canvas
        c.delete("all")
        width = c.winfo_width() or 600
        height = c.winfo_height() or 680

        self._draw_orb(c, width, height)
        self._draw_button(c, width, height)
        self._draw_chat_bubble(c, width, height)

    def _draw_orb(self, c, width, height):
        cx, cy = width / 2, height * 0.30
        base_radius = min(width, height) * 0.22
        pulse = math.sin(self._pulse_t) * self.STATE_PULSE.get(self.state, 0.0)
        radius = base_radius + pulse

        segments = 140
        stroke_width = max(4, int(base_radius * 0.045))

        for i in range(segments):
            t0 = i / segments
            t1 = (i + 1.4) / segments
            a0 = 2 * math.pi * t0 + self._phase * 2 * math.pi
            a1 = 2 * math.pi * t1 + self._phase * 2 * math.pi
            x0, y0 = cx + radius * math.cos(a0), cy + radius * math.sin(a0)
            x1, y1 = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
            color = _gradient_color(t0)
            c.create_line(x0, y0, x1, y1, fill=color, width=stroke_width, capstyle="round")

        inner_r = radius * 0.7
        c.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="#1a1a20", width=1)

    def _draw_button(self, c, width, height):
        label = "End Conversation" if self.is_active else "Start Conversation"
        btn_w, btn_h = max(260, width * 0.5), 54
        cx, cy = width / 2, height * 0.66
        x1, y1 = cx - btn_w / 2, cy - btn_h / 2
        x2, y2 = cx + btn_w / 2, cy + btn_h / 2

        round_rectangle(c, x1, y1, x2, y2, radius=10, outline="#f2f2f2", fill="", width=2)

        font = ("Segoe UI", 14, "bold")
        temp = c.create_text(0, 0, text=label, font=font)
        bbox = c.bbox(temp)
        c.delete(temp)
        text_w = (bbox[2] - bbox[0]) if bbox else 160

        icon_r = 7
        gap = 12
        group_w = icon_r * 2 + gap + text_w
        group_x1 = cx - group_w / 2

        icon_cx = group_x1 + icon_r
        c.create_oval(icon_cx - icon_r, cy - icon_r, icon_cx + icon_r, cy + icon_r, outline="#f2f2f2", width=2)

        text_x = group_x1 + icon_r * 2 + gap
        c.create_text(text_x, cy, text=label, fill="#f2f2f2", font=font, anchor="w")
        self._end_button_bbox = (x1, y1, x2, y2)

    def _draw_chat_bubble(self, c, width, height):
        who, text = self.last_message
        pad_x = 18
        max_bubble_w = width * 0.7
        font = ("Segoe UI", 13)

        temp = c.create_text(0, 0, text=text, font=font, width=max_bubble_w - 2 * pad_x)
        bbox = c.bbox(temp)
        c.delete(temp)
        text_w = (bbox[2] - bbox[0]) if bbox else 60
        text_h = (bbox[3] - bbox[1]) if bbox else 20

        bubble_w = text_w + 2 * pad_x
        bubble_h = text_h + pad_x
        avatar_r = 14
        avatar_gap = 14

        x2 = width - pad_x - avatar_r * 2 - avatar_gap
        x1 = x2 - bubble_w
        y2 = height - pad_x
        y1 = y2 - bubble_h

        bubble_fill = "#26272e" if who == "assistant" else "#1c3a2e"
        round_rectangle(c, x1, y1, x2, y2, radius=bubble_h / 2, fill=bubble_fill, outline="")
        c.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill="#f2f2f2", font=font, width=max_bubble_w - 2 * pad_x)

        acx = x2 + avatar_gap + avatar_r
        acy = (y1 + y2) / 2
        c.create_oval(acx - avatar_r * 0.45, acy - avatar_r * 0.75, acx + avatar_r * 0.45, acy - avatar_r * 0.05,
                       outline="#9a9ea8", width=2)
        c.create_arc(acx - avatar_r, acy + avatar_r * 0.9, acx + avatar_r, acy - avatar_r * 0.2,
                     start=0, extent=180, style="arc", outline="#9a9ea8", width=2)
    

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------
    def _animate(self):
        speed = self.STATE_SPEED.get(self.state, 0.6)
        self._phase = (self._phase + 0.0015 * speed) % 1.0
        self._pulse_t += 0.12
        self._draw()
        self.after(30, self._animate)

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if self._end_button_bbox:
            x1, y1, x2, y2 = self._end_button_bbox
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                if self.is_active:
                    self.is_active = False
                    self.on_stop_session()  # HOOK 2
                else:
                    self.is_active = True
                    self.on_start_session()  # HOOK 1

    # ------------------------------------------------------------------
    # HOOK 1 & 2 — wire up your Deepgram start/stop logic here.
    # Use self.get_settings_json() to get the Settings payload to send.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def on_start_session(self):
        """Override this method to start the Deepgram session."""
        try:
            self.agent_connection, self.agent_connection_cm = start_agent_connection(self._msg_queue)
            self.mic_running = True

            self.mic_running = True
            self.mic_thread = threading.Thread(target=self._mic_capture_loop, daemon=True)
            self.mic_thread.start()
            print("Deepgram agent connection started.")
            print("Microphone started.") 
        except Exception as e:
            print(f"Failed to start Deepgram agent connection: {e}")
            self.is_active = False
    def _connect_agent(self):
        try:
            self.connection, self.connection_cm = start_agent_connection(self._msg_queue)
            
            self.mic_thread = threading.Thread(target=self._mic_capture_loop, daemon=True)
            self.mic_thread.start()
        
        except Exception as e:
            self._msg_queue.put(("error", f"Failed to connect to Deepgram agent: {e}"))

            self.is_running = False
            self.is_active = False
    def _audio_playback_loop(self):
        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=self.output_sample_rate.get(), output=True)
            while true:
                audio_data = self.audio_queue
                if audio_data is None:
                    break
                stream.write(audio_data)
        except Exception as e:
            self._msg_queue.put(("error", f"Audio playback error: {e}"))
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self._msg_queue.put(("log", ("system", "Audio playback stopped.")))
    def _mic_capture_loop(self):
        if not getattr(self, "agent_connection", None):
            print("No Deepgram connection.")
            return

        sample_rate = self.input_sample_rate.get()
        channels = 1
        blocksize = 4800

        print("Microphone capture started.")

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=blocksize,
                channels=channels,
                dtype="int16",
                callback=self._mic_callback
            ):
                while getattr(self, "mic_running", False):
                    sd.sleep(100)

        except Exception as e:
            print(f"Microphone error: {e}")

        print("Microphone capture stopped.")
    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(f"Microphone status: {status}")
        if not getattr(self, "mic_running", False):
            return
        connection = getattr(self, "agent_connection", None)
        if connection is None:
            return
        try:
            audio_bytes = bytes(indata)

            if audio_bytes:
                send_audio_chunk(self.agent_connection, audio_bytes)
        except Exception as e:
            print(f"Error sending audio chunk: {e}")
    def on_stop_session(self):
        self.mic_running = False

        if getattr(self, "agent_connection_cm", None):
            try:
                stop_agent_connection(self.agent_connection_cm)
                print("Deepgram agent connection stopped.")
            except Exception as e:
                print(f"Error stopping Deepgram agent connection: {e}")
        self.agent_connection = None
        self.agent_connection_cm = None
        print("Microphone stopped.")
    
    def _append_message(self, who, text):
        self.last_message = (who, text)
        self._append_transcript(who, text)
    
    def _update_orb_state(self, new_state):
        self.state = new_state
        self.status_note.configure(text=f"Status: {new_state.capitalize()}")
    
    def _show_interim_text(self, who, text):
        self.last_message = (who, text)
        self._draw()
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(*payload)          # your log widget
                elif kind == "message":
                    self._append_message(*payload)       # your chat bubble
                elif kind == "state":
                    self._update_orb_state(payload)       # your orb animation
                elif kind == "interim":
                    self._show_interim_text(*payload)
                elif kind == "status":
                    self._append_log("system", str(payload))
                elif kind == "error":
                    self._append_log("system", f"Error: {payload}")
                elif kind == "audio":
                    self.audio_queue.put(payload)  # send audio data to playback queue
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)


