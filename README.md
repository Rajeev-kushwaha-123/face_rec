# Face Recognition System

A face recognition system built with Python, OpenCV, and Tkinter that allows users to detect and recognize faces in images.

## Features

- Upload and process images containing faces
- Detect faces in images using OpenCV
- Extract face features using face_recognition library
- Add new faces to the known faces database
- Recognize faces with confidence scores
- Simple and intuitive GUI interface

## Requirements

- Python 3.7 or higher
- OpenCV
- face_recognition
- NumPy
- Pillow
- Tkinter (usually comes with Python)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python face_recognition_app.py
```

2. Using the application:
   - Click "Upload Image" to select an image containing faces
   - Use "Add New Face" to add a face to the known faces database
   - Click "Recognize Faces" to detect and recognize faces in the current image
   - Results will be displayed in the text area below the image

## How it Works

1. Face Detection: The system uses the face_recognition library to detect faces in images
2. Feature Extraction: Face features are extracted using the face_recognition library's encoding function
3. Face Recognition: Detected faces are compared against the known faces database
4. Results: The system displays the recognized faces with confidence scores

## Notes

- The known faces database is stored in `known_faces.json`
- Images should be clear and well-lit for better recognition results
- The system works best with frontal face images 