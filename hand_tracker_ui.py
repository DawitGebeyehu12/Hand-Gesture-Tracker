import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
from pathlib import Path
import threading
from typing import Optional
import os
from HandTracker import HandTracker
import queue

class ModernHandTrackerUI:
    def __init__(self):
        # Set theme and color scheme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize main window
        self.window = ctk.CTk()
        self.window.title("Hand Gesture Tracker")
        self.window.geometry("1280x720")
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # Create main container
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)

        # Create header
        self.create_header()

        # Create content area
        self.create_content_area()

        # Create footer
        self.create_footer()

        # Initialize variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.tracker: Optional[HandTracker] = None
        self.is_webcam_active = False
        self.video_thread: Optional[threading.Thread] = None
        self.processed_video_path: Optional[str] = None

        # Add a queue for frame updates
        self.frame_queue = queue.Queue()
        
        # Start the update loop
        self.update_loop()

    def create_header(self):
        """Create the header section with title and description"""
        header = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(0, 20), sticky="ew")

        title = ctk.CTkLabel(
            header,
            text="Hand Gesture Tracker",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 10))

        description = ctk.CTkLabel(
            header,
            text="Upload a video file or use your webcam to track hand gestures in real-time",
            font=ctk.CTkFont(size=14)
        )
        description.pack()

    def create_content_area(self):
        """Create the main content area with video display and controls"""
        content = ctk.CTkFrame(self.main_container)
        content.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # Controls section
        controls = ctk.CTkFrame(content, fg_color="transparent")
        controls.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # File upload section
        self.upload_frame = ctk.CTkFrame(controls, fg_color="#2a2d2e")
        self.upload_frame.pack(side="left", padx=(0, 10), fill="x", expand=True)

        upload_label = ctk.CTkLabel(
            self.upload_frame,
            text="Upload Video File",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        upload_label.pack(pady=(10, 5))

        self.file_label = ctk.CTkLabel(
            self.upload_frame,
            text="No file selected",
            font=ctk.CTkFont(size=12)
        )
        self.file_label.pack(pady=(0, 5))

        upload_button = ctk.CTkButton(
            self.upload_frame,
            text="Choose File",
            command=self.upload_file,
            font=ctk.CTkFont(size=13)
        )
        upload_button.pack(pady=(0, 10))

        # Webcam control section
        self.webcam_frame = ctk.CTkFrame(controls, fg_color="#2a2d2e")
        self.webcam_frame.pack(side="right", padx=(10, 0), fill="x", expand=True)

        webcam_label = ctk.CTkLabel(
            self.webcam_frame,
            text="Webcam Control",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        webcam_label.pack(pady=(10, 5))

        self.webcam_button = ctk.CTkButton(
            self.webcam_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            font=ctk.CTkFont(size=13)
        )
        self.webcam_button.pack(pady=(5, 10))

        # Video display area
        self.video_frame = ctk.CTkFrame(content)
        self.video_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True)

    def create_footer(self):
        """Create the footer section with status and credits"""
        footer = ctk.CTkFrame(self.main_container, fg_color="transparent")
        footer.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.status_label = ctk.CTkLabel(
            footer,
            text="Ready to start",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left")

        credits = ctk.CTkLabel(
            footer,
            text="Hand Gesture Tracker v1.0",
            font=ctk.CTkFont(size=12)
        )
        credits.pack(side="right")

    def upload_file(self):
        """Handle file upload"""
        file_types = (("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        file_path = ctk.filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            self.file_label.configure(text=os.path.basename(file_path))
            self.status_label.configure(text="Processing video...")
            
            # Stop webcam if it's running
            if self.is_webcam_active:
                self.toggle_webcam()

            # Start video processing in a separate thread
            self.video_thread = threading.Thread(
                target=self.process_video,
                args=(file_path,)
            )
            self.video_thread.daemon = True
            self.video_thread.start()

    def process_video(self, file_path: str):
        """Process the uploaded video file"""
        try:
            self.cap = cv2.VideoCapture(file_path)
            self.tracker = HandTracker()
            
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                processed_frame = self.tracker.process_frame(frame)
                self.update_video_display(processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            self.status_label.configure(text="Video processing completed")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")

    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.is_webcam_active:
            self.webcam_button.configure(text="Stop Webcam")
            self.status_label.configure(text="Webcam active")
            self.is_webcam_active = True
            
            # Start webcam in a separate thread
            self.video_thread = threading.Thread(target=self.process_webcam)
            self.video_thread.daemon = True
            self.video_thread.start()
        else:
            self.webcam_button.configure(text="Start Webcam")
            self.status_label.configure(text="Webcam stopped")
            self.is_webcam_active = False
            
            if self.cap:
                self.cap.release()

    def process_webcam(self):
        """Process webcam feed"""
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        
        while self.is_webcam_active:
            success, frame = self.cap.read()
            if not success:
                break

            processed_frame = self.tracker.process_frame(frame)
            self.update_video_display(processed_frame)

        self.cap.release()

    def update_loop(self):
        """Check for new frames and update the display"""
        try:
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                
                # Convert and display the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(frame)
                photo = PIL.ImageTk.PhotoImage(image=image)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
        except queue.Empty:
            pass
            
        # Schedule the next update
        self.window.after(10, self.update_loop)

    def update_video_display(self, frame):
        """Queue the frame for display"""
        self.frame_queue.put(frame)

    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = ModernHandTrackerUI()
    app.run()