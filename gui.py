import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


class FaceApp:
    def __init__(self, detector, embedder, matcher, database):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.database = database

        # آخر وجه تم التقاطه (نفس المتغيرات العالمية في aa.py)
        self.last_processed_face = None
        self.last_embedding = None

        self.camera = cv2.VideoCapture(0)

        self.root = tk.Tk()
        self.root.title("Smart Facial Recognition System")
        self.root.geometry("700x650")
        self.root.configure(bg="#2c3e50")

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        input_frame = tk.Frame(self.root, bg="#2c3e50")
        input_frame.pack(pady=10)

        self.name_entry = tk.Entry(input_frame, font=("Arial", 16), width=20, justify="center")
        self.name_entry.insert(0, "Write the name here...")
        self.name_entry.pack(side=tk.RIGHT, padx=10)
        self.name_entry.bind("<FocusIn>", self.clear_placeholder)

        save_btn = tk.Button(input_frame, text="Save the new person",
                             font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
                             command=self.save_new_person)
        save_btn.pack(side=tk.LEFT, padx=10)

    def clear_placeholder(self, event):
        if self.name_entry.get() == "Write the name here...":
            self.name_entry.delete(0, tk.END)

    def save_new_person(self):
        name = self.name_entry.get().strip()

        if name == "" or name == "Write the name here...":
            messagebox.showwarning("Warning", "Please write the name first.")
            return

        if self.last_processed_face is not None and self.last_embedding is not None:
            self.database.save_person(name, self.last_processed_face, self.last_embedding)
            messagebox.showinfo("Success", f"[{name}] has been successfully added to the database!")
            self.name_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Error: Face not detected by the camera!")

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            faces = self.detector.detect(frame)

            for (x, y, w, h) in faces:
                face_roi = self.embedder.crop(frame, (x, y, w, h))

                if face_roi is not None and face_roi.size > 0:
                    resized_face = self.embedder.preprocess(face_roi)
                    live_embedding = self.embedder.embed(face_roi)

                    identity, distance = self.matcher.recognize(
                        live_embedding, self.database.known_embeddings)

                    if identity != "Unknown":
                        color = (0, 255, 0)
                        status_text = f"Access Granted: {identity}"
                    else:
                        color = (0, 0, 255)
                        status_text = "Access Denied: Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{status_text} ({distance:.2f})", (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    self.last_processed_face = resized_face
                    self.last_embedding = live_embedding

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        self.root.after(10, self.update_frame)

    def run(self):
        self.update_frame()
        self.root.mainloop()
        self.camera.release()
        cv2.destroyAllWindows()