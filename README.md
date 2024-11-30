Hand Gesture Tracker

The Hand Gesture Tracker is a modern application built using customtkinter, OpenCV, 
and Mediapipe that enables real-time hand gesture recognition. Users can upload video
files or utilize a webcam to detect and track hand movements and gestures efficiently.
The application provides a user-friendly interface and is a robust tool for exploring hand 
tracking capabilities.

Features

- Real-Time Hand Tracking: Leverages Mediapipe's hand tracking capabilities for accurate and efficient detection.

- Video File Processing: Supports the upload and processing of video files with hand gestures.

- Webcam Support: Enables real-time tracking directly from a webcam.

- Modern UI: Implements a sleek, intuitive interface using customtkinter.
  
- Gesture Information Overlay: Displays gesture names and confidence levels on the video feed.
  
- Responsive Performance: Utilizes threading and a queue-based frame update system for smooth UI and video processing.


Prerequisites
Before using the application, ensure the following dependencies are installed:

Python 3.7 or higher
Required libraries:

     pip install customtkinter opencv-python-headless mediapipe pillow numpy

Installation

1. Clone the repository:
   
       git clone https://github.com/your-username/hand-gesture-tracker.git
       cd hand-gesture-tracker

2. Run the application:

       python app.py

Usage

1. Launch the Application: Run app.py and the main window will appear.

2. Upload a Video File:

    - Click on the "Choose File" button under the "Upload Video File" section.

    - Select a video file (supported formats: .mp4, .avi, .mov).
     
    - The app will process the video and display tracked gestures.

3. Enable Webcam:

    - Click on "Start Webcam" under the "Webcam Control" section.

    - Real-time tracking will begin.

4. Stop Webcam:

    - Click "Stop Webcam" to disable real-time tracking.
