import tkinter as tk
import subprocess
import sys
import os

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom Vision Apps")
        self.root.geometry("400x500")
        self.root.configure(bg="#222222")

        self.label = tk.Label(root, text="Select an App:", font=("Arial", 20), bg="#222222", fg="white")
        self.label.pack(pady=20)

        self.maths_btn = tk.Button(
            root,
            text="Finger Math",
            font=("Arial", 16),
            bg="white",
            fg="black",
            activebackground="#cccccc",
            activeforeground="black",
            width=15,
            height=2,
            borderwidth=2,
            relief="solid",
            command=self.launch_finger_counter
        )
        self.maths_btn.pack(pady=10)

        self.track_btn = tk.Button(
            root,
            text="Finger Track",
            font=("Arial", 16),
            bg="white",
            fg="black",
            activebackground="#cccccc",
            activeforeground="black",
            width=15,
            height=2,
            borderwidth=2,
            relief="solid",
            command=self.launch_tracker
        )
        self.track_btn.pack(pady=10)

        self.vanish_btn = tk.Button(
            root,
            text="Vanish Shirt",
            font=("Arial", 16),
            bg="white",
            fg="black",
            activebackground="#cccccc",
            activeforeground="black",
            width=15,
            height=2,
            borderwidth=2,
            relief="solid",
            command=self.launch_vanisher
        )
        self.vanish_btn.pack(pady=10)

        self.body_btn = tk.Button(
            root,
            text="Body Detector",
            font=("Arial", 16),
            bg="white",
            fg="black",
            activebackground="#cccccc",
            activeforeground="black",
            width=15,
            height=2,
            borderwidth=2,
            relief="solid",
            command=self.launch_body_detector
        )
        self.body_btn.pack(pady=10)

        # Keep track of processes
        self.math_process = None
        self.track_process = None
        self.vanish_process = None
        self.body_process = None

    def launch_finger_counter(self):
        if self.math_process is not None and self.math_process.poll() is None:
            print("Finger Math is already running.")
            return
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "finger_counter.py")
        self.math_process = subprocess.Popen([python_exe, script_path])

    def launch_tracker(self):
        if self.track_process is not None and self.track_process.poll() is None:
            print("Finger Tracker is already running.")
            return
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "finger_tracker.py")
        self.track_process = subprocess.Popen([python_exe, script_path])

    def launch_vanisher(self):
        if self.vanish_process is not None and self.vanish_process.poll() is None:
            print("Vanish Shirt is already running.")
            return
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "vanishing_shirt.py")
        self.vanish_process = subprocess.Popen([python_exe, script_path])

    def launch_body_detector(self):
        if self.body_process is not None and self.body_process.poll() is None:
            print("Body Detector is already running.")
            return
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "body_detector.py")
        self.body_process = subprocess.Popen([python_exe, script_path])

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop() 