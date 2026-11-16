import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from torchvision import transforms

BASE = Path(__file__).resolve().parent
CAPTURES = BASE / "captures"


class AgeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.30),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class FaceCist:
    def __init__(self, root):
        self.root = root
        root.title("FaceCist | Face Detection & Age Estimation")
        root.geometry("1100x760")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = False
        self.cap = None
        self.lock = threading.Lock()
        self.latest = None
        self.photo = None
        self.result_photo = None
        self.boxes, self.ages = [], []
        self.frame_no = 0
        self.last_t = time.perf_counter()
        self.fps = 0.0
        self.interval = 4  # infer every fourth camera frame to reduce CPU load

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)), transforms.ToTensor(),
            transforms.Normalize([.5]*3, [.5]*3)
        ])
        self._ui()
        self._load_models()
        self.refresh_results()
        root.bind_all("<space>", self.space_capture)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _ui(self):
        head = ttk.Frame(self.root, padding=14)
        head.pack(fill="x")
        ttk.Label(head, text="FaceCist", font=("Segoe UI", 22, "bold")).pack(side="left")
        self.status = tk.StringVar(value="Loading models…")
        ttk.Label(head, textvariable=self.status).pack(side="right")

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        live = ttk.Frame(self.tabs, padding=12)
        results = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(live, text="  Live Detection  ")
        self.tabs.add(results, text="  Results  ")

        bar = ttk.Frame(live)
        bar.pack(fill="x", pady=(0, 10))
        self.start_btn = ttk.Button(bar, text="Start Detecting", command=self.toggle, state="disabled")
        self.start_btn.pack(side="left")
        ttk.Button(bar, text="Capture Screenshot  (Space)", command=self.capture).pack(side="left", padx=8)
        self.info = tk.StringVar(value="Press Start Detecting to open your webcam.")
        ttk.Label(bar, textvariable=self.info).pack(side="right")
        self.video = ttk.Label(live, text="Camera preview will appear here",
                               anchor="center", background="#171717", foreground="white")
        self.video.pack(fill="both", expand=True)

        split = ttk.Frame(results)
        split.pack(fill="both", expand=True)
        left = ttk.Frame(split)
        left.pack(side="left", fill="y", padx=(0, 12))
        ttk.Label(left, text="Captured screenshots", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=38, height=28, exportselection=False)
        self.listbox.pack(fill="y", expand=True, pady=8)
        self.listbox.bind("<<ListboxSelect>>", self.show_result)
        ttk.Button(left, text="Refresh", command=self.refresh_results).pack(fill="x")
        self.preview = ttk.Label(split, text="Select a screenshot to preview it.",
                                 anchor="center", background="#171717", foreground="white")
        self.preview.pack(side="left", fill="both", expand=True)
        ttk.Label(self.root, text=f"Device: {self.device}  •  Saved screenshots: {CAPTURES}",
                  padding=(14, 0, 14, 10)).pack(fill="x")

    def _load_models(self):
        yolo, age = BASE / "yolov8_face.pt", BASE / "age_regressor.pth"
        missing = [x.name for x in (yolo, age) if not x.is_file()]
        if missing:
            messagebox.showerror("Missing model files",
                "Place these files beside webcam_age_gui.py:\n\n" + "\n".join(missing))
            self.status.set("Model files missing")
            return
        try:
            self.status.set(f"Loading models on {self.device}…")
            self.root.update_idletasks()
            self.detector = YOLO(str(yolo))
            self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            for param in self.facenet.parameters():
                param.requires_grad_(False)
            self.age_model = AgeRegressor().eval().to(self.device)
            self.age_model.load_state_dict(torch.load(age, map_location=self.device, weights_only=True))
            self.start_btn.configure(state="normal")
            self.status.set(f"Models ready • {self.device}")
        except Exception as exc:
            self.status.set("Model loading failed")
            messagebox.showerror("Model loading failed", str(exc))

    def toggle(self):
        if self.running:
            self.stop()
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror("Webcam error", "Could not open camera 0. Close other apps using it.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.start_btn.configure(text="Stop Detecting")
        self.status.set("Webcam active")
        self.info.set("SPACE = save screenshot")
        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.update_preview()

    def camera_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(.02)
                continue
            frame = cv2.flip(frame, 1)
            self.frame_no += 1
            if self.frame_no % self.interval == 1:
                try:
                    pred = self.detector.predict(frame, conf=.20, imgsz=416,
                        device=0 if self.device.type == "cuda" else "cpu", verbose=False)[0]
                    new_boxes, new_ages, tensors, items = [], [], [], []
                    if pred.boxes is not None and len(pred.boxes):
                        h, w = frame.shape[:2]
                        for box, score in zip(pred.boxes.xyxy.cpu().numpy(),
                                              pred.boxes.conf.cpu().numpy()):
                            x1, y1, x2, y2 = map(int, box)
                            bw, bh = x2-x1, y2-y1
                            if bw < 2 or bh < 2:
                                continue
                            xa, ya = max(0, x1-int(bw*.08)), max(0, y1-int(bh*.08))
                            xb, yb = min(w, x2+int(bw*.08)), min(h, y2+int(bh*.08))
                            rgb = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2RGB)
                            tensors.append(self.transform(Image.fromarray(rgb)))
                            items.append((x1, y1, x2, y2, float(score)))
                    if tensors:
                        with torch.inference_mode():
                            embeddings = self.facenet(torch.stack(tensors).to(self.device))
                            ages = self.age_model(embeddings).clamp(0, 116).cpu().tolist()
                        for item, age in zip(items, ages):
                            new_boxes.append(item)
                            new_ages.append(float(age))
                    self.boxes, self.ages = new_boxes, new_ages
                except Exception as exc:
                    self.root.after(0, lambda err=str(exc): self.status.set(f"Inference error: {err}"))

            shown = frame.copy()
            for (x1, y1, x2, y2, score), age in zip(self.boxes, self.ages):
                cv2.rectangle(shown, (x1,y1), (x2,y2), (50,220,90), 2)
                text = f"Age ~ {age:.0f} | face {score:.2f}"
                cv2.putText(shown, text, (x1,max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(shown, text, (x1,max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (20,90,20), 1, cv2.LINE_AA)
            now = time.perf_counter()
            inst = 1 / max(now-self.last_t, 1e-6)
            self.last_t = now
            self.fps = inst if not self.fps else .85*self.fps + .15*inst
            cv2.putText(shown, f"FPS: {self.fps:.1f} | Faces: {len(self.boxes)}",
                        (12,26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,255,255), 2)
            with self.lock:
                self.latest = shown

    def update_preview(self):
        if not self.running:
            return
        with self.lock:
            frame = None if self.latest is None else self.latest.copy()
        if frame is not None:
            self.display_image(frame, self.video, "live")
        self.root.after(30, self.update_preview)

    def display_image(self, frame, widget, which):
        maxw, maxh = max(widget.winfo_width(), 500), max(widget.winfo_height(), 350)
        h, w = frame.shape[:2]
        scale = min(maxw/w, maxh/h)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        widget.configure(image=image, text="")
        if which == "live":
            self.photo = image
        else:
            self.result_photo = image

    def space_capture(self, event=None):
        if self.running and self.tabs.index(self.tabs.select()) == 0:
            self.capture()
            return "break"

    def capture(self):
        if not self.running:
            self.info.set("Start detecting before taking a screenshot.")
            return
        with self.lock:
            frame = None if self.latest is None else self.latest.copy()
        if frame is None:
            return
        CAPTURES.mkdir(exist_ok=True)
        filename = time.strftime("FaceCist_%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}.jpg"
        cv2.imwrite(str(CAPTURES / filename), frame)
        self.info.set(f"Saved: {filename}")
        self.refresh_results()
        self.status.set(f"Screenshot saved • {filename}")

    def refresh_results(self):
        CAPTURES.mkdir(exist_ok=True)
        self.paths = sorted(CAPTURES.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.listbox.delete(0, tk.END)
        for path in self.paths:
            self.listbox.insert(tk.END, path.name)
        if not self.paths:
            self.preview.configure(image="", text="No screenshots yet. Capture one from Live Detection.")
            self.result_photo = None

    def show_result(self, event=None):
        sel = self.listbox.curselection()
        if sel:
            frame = cv2.imread(str(self.paths[sel[0]]))
            if frame is not None:
                self.display_image(frame, self.preview, "result")

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(text="Start Detecting")
        self.status.set(f"Models ready • {self.device}")
        self.info.set("Detection stopped. Press Start Detecting to resume.")

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    FaceCist(app_root)
    app_root.mainloop()
