import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import pickle

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')  # Light gray background
        
        # Initialize variables
        self.current_image = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create left panel for controls
        left_panel = ttk.Frame(main_container, padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        # Style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10, font=('Arial', 10))
        
        # Title
        title_label = ttk.Label(left_panel, text="Face Recognition System", 
                              font=("Arial", 16, "bold"), padding=(0, 0, 0, 20))
        title_label.pack(fill="x")
        
        # Control buttons
        ttk.Button(left_panel, text="Upload Image", command=self.upload_image,
                  style='Custom.TButton').pack(fill="x", pady=5)
        ttk.Button(left_panel, text="Add New Face", command=self.add_new_face,
                  style='Custom.TButton').pack(fill="x", pady=5)
        ttk.Button(left_panel, text="Recognize Faces", command=self.recognize_faces,
                  style='Custom.TButton').pack(fill="x", pady=5)
        
        # Known faces list
        ttk.Label(left_panel, text="Known Faces", font=("Arial", 12, "bold"),
                 padding=(0, 20, 0, 10)).pack(fill="x")
        
        self.faces_listbox = tk.Listbox(left_panel, height=10, font=("Arial", 10))
        self.faces_listbox.pack(fill="x", pady=5)
        self.update_faces_list()
        
        # Create right panel for image and results
        right_panel = ttk.Frame(main_container, padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_rowconfigure(0, weight=0)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Image frame with fixed size
        image_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding="10")
        image_frame.grid(row=0, column=0, sticky="nsew")
        image_frame.config(width=650, height=420)
        image_frame.grid_propagate(False)
        
        self.image_label = ttk.Label(image_frame, background="#e0e0e0", anchor="center")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center", width=600, height=400)
        
        # Results frame with border, always visible below image
        results_frame = ttk.LabelFrame(right_panel, text="Recognition Results", padding="10")
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        results_frame.config(height=200)
        results_frame.grid_propagate(False)
        
        # Create a canvas with scrollbar for results
        results_canvas = tk.Canvas(results_frame, height=180, bg="#f9f9f9", highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_canvas.yview)
        self.results_text = tk.Text(results_canvas, height=8, width=50, font=("Arial", 11),
                                  yscrollcommand=scrollbar.set, bg="#f9f9f9", relief="flat", borderwidth=0)
        
        # Pack the scrollbar and text widget
        scrollbar.pack(side="right", fill="y")
        results_canvas.pack(side="left", fill="both", expand=True)
        results_canvas.create_window((0, 0), window=self.results_text, anchor="nw")
        self.results_text.bind("<Configure>", lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all")))
        
        # Configure grid weights for main container
        main_container.grid_columnconfigure(1, weight=3)
        main_container.grid_rowconfigure(0, weight=1)
        
    def update_faces_list(self):
        self.faces_listbox.delete(0, tk.END)
        for name in self.known_face_names:
            self.faces_listbox.insert(tk.END, name)
        
    def load_known_faces(self):
        if os.path.exists("known_faces.pkl"):
            with open("known_faces.pkl", "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
                if len(self.known_face_encodings) > 0:
                    self.face_recognizer.train(self.known_face_encodings, np.array(range(len(self.known_face_encodings))))
    
    def save_known_faces(self):
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        with open("known_faces.pkl", "wb") as f:
            pickle.dump(data, f)
        self.update_faces_list()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
    
    def display_image(self, image):
        if image is None:
            return
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit in 600x400 area
        max_width, max_height = 600, 400
        height, width = image_rgb.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        new_width, new_height = int(width * scale), int(height * scale)
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def add_new_face(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Please upload an image first")
            return
            
        # Get name for the new face
        name = simpledialog.askstring("Input", "Enter name for the face:")
        if not name:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected in the image")
            return
            
        # Get the first face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Add to known faces
        self.known_face_encodings.append(face_roi)
        self.known_face_names.append(name)
        
        # Retrain the recognizer
        if len(self.known_face_encodings) > 0:
            self.face_recognizer.train(self.known_face_encodings, np.array(range(len(self.known_face_encodings))))
        
        # Save to file
        self.save_known_faces()
        
        messagebox.showinfo("Success", f"Added {name} to known faces")
    
    def recognize_faces(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Please upload an image first")
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected in the image")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Recognize each face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            if len(self.known_face_encodings) > 0:
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    if confidence < 100:  # Lower confidence is better in LBPH
                        name = self.known_face_names[label]
                        self.results_text.insert(tk.END, f"Found {name} (Confidence: {100-confidence:.2f}%)\n")
                    else:
                        self.results_text.insert(tk.END, "Unknown face detected\n")
                except:
                    self.results_text.insert(tk.END, "Error recognizing face\n")
            else:
                self.results_text.insert(tk.END, "No known faces in database\n")
            
            # Draw rectangle around face
            cv2.rectangle(self.current_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Update display
        self.display_image(self.current_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop() 