import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Object Detection")

        # Initialize variables
        self.video_source = 0  # Default webcam
        self.cap = None
        self.thread = None
        self.running = False
        self.current_image_path = None
        self.detection_enabled = True

        # Load Haar Cascade Classifiers
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eyeglass_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.eyes_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.fullbody_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

        # Create GUI
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

        # Controls frame
        self.controls_frame = ttk.Frame(self.main_frame, padding="10")
        self.controls_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))

        # Buttons
        self.start_btn = ttk.Button(self.controls_frame, text="Start", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)

        self.stop_btn = ttk.Button(self.controls_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)

        self.browse_btn = ttk.Button(self.controls_frame, text="Browse Image", command=self.browse_image)
        self.browse_btn.grid(row=0, column=2, padx=5, pady=5)

        self.toggle_detection_btn = ttk.Button(self.controls_frame, text="Toggle Detection",
                                               command=self.toggle_detection)
        self.toggle_detection_btn.grid(row=0, column=3, padx=5, pady=5)

        # Detection options
        self.detection_options_frame = ttk.LabelFrame(self.main_frame, text="Detection Options", padding="10")
        self.detection_options_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E))

        self.face_var = tk.IntVar(value=1)
        self.eye_var = tk.IntVar(value=1)
        self.eyeglass_var = tk.IntVar(value=1)
        self.body_var = tk.IntVar(value=1)

        ttk.Checkbutton(self.detection_options_frame, text="Faces", variable=self.face_var).grid(row=0, column=0,
                                                                                                 padx=5, sticky=tk.W)
        ttk.Checkbutton(self.detection_options_frame, text="Eyes", variable=self.eye_var).grid(row=0, column=1, padx=5,
                                                                                               sticky=tk.W)
        ttk.Checkbutton(self.detection_options_frame, text="Eyeglasses", variable=self.eyeglass_var).grid(row=0,
                                                                                                          column=2,
                                                                                                          padx=5,
                                                                                                          sticky=tk.W)
        ttk.Checkbutton(self.detection_options_frame, text="Full Body", variable=self.body_var).grid(row=0, column=3,
                                                                                                     padx=5,
                                                                                                     sticky=tk.W)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.DISABLED)

            # Release any existing capture
            if self.cap is not None:
                self.cap.release()

            # Initialize video capture
            self.cap = cv2.VideoCapture(self.video_source)

            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video source")
                self.stop_detection()
                return

            self.status_var.set("Detection running...")
            self.thread = threading.Thread(target=self.update_frame, daemon=True)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.NORMAL)
        self.status_var.set("Ready")

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        status = "ON" if self.detection_enabled else "OFF"
        self.status_var.set(f"Detection {status}")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.current_image_path = file_path
            self.process_image(file_path)

    def process_image(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            messagebox.showerror("Error", "Could not load the image")
            return

        if self.detection_enabled:
            img = self.detect_objects(img)

        self.display_image(img)

    def detect_objects(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        if self.face_var.get():
            faces = self.face_detector.detectMultiScale(img_gray, minNeighbors=20)
            for (x1, y1, w, h) in faces:
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

                # Eye and eyeglass detection within face region
                face_region = img_gray[y1:y1 + h, x1:x1 + w]

                if self.eye_var.get():
                    eyes = self.eyes_detector.detectMultiScale(face_region)
                    for (x1e, y1e, we, he) in eyes:
                        cv2.rectangle(img, (x1 + x1e, y1 + y1e), (x1 + x1e + we, y1 + y1e + he), (255, 0, 0), 2)

                if self.eyeglass_var.get():
                    eyeglasses = self.eyeglass_detector.detectMultiScale(face_region)
                    for (x1g, y1g, wg, hg) in eyeglasses:
                        cv2.rectangle(img, (x1 + x1g, y1 + y1g), (x1 + x1g + wg, y1 + y1g + hg), (0, 0, 255), 2)

        # Full body detection
        if self.body_var.get():
            full_bodies = self.fullbody_detector.detectMultiScale(img_gray, minNeighbors=20)
            for (x1b, y1b, wb, hb) in full_bodies:
                cv2.rectangle(img, (x1b, y1b), (x1b + wb, y1b + hb), (0, 255, 255), 2)

        return img

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                self.status_var.set("Error reading from video source")
                break

            if self.detection_enabled:
                frame = self.detect_objects(frame)

            self.display_image(frame)

            # Small delay to prevent high CPU usage
            cv2.waitKey(30)

    def display_image(self, img):
        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to fit the window while maintaining aspect ratio
        h, w = img_rgb.shape[:2]
        max_width = self.root.winfo_width() - 20
        max_height = 500  # Max height for display

        if w > max_width:
            ratio = max_width / w
            new_w = max_width
            new_h = int(h * ratio)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        if new_h > max_height:
            ratio = max_height / new_h
            new_h = max_height
            new_w = int(new_w * ratio)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        # Convert to PhotoImage
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Update the label
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # Keep a reference

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()