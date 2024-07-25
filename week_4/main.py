import tkinter as tk
from PIL import Image, ImageTk
import util
import cv2
import os
import subprocess  # Import subprocess module
import datetime

class App():
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x800")

        self.login_button_main_window = util.get_button(self.main_window, 'login', "green", self.login)
        self.login_button_main_window.place(x=750, y=300)
        self.register_new_user_button_main_window = util.get_button(self.main_window, 'New User', "red", self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        # Webcam
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=800, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):  # Fix typo: check if directory does not exist
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'  # Initialize log_path

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        
        self.label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_arr = frame
            self.most_recent_capture_pil = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Schedule the next update in 20 milliseconds
        self.main_window.after(20, self.process_webcam)

    def login(self):
        unknown_img_path = './.tmp.jpg'  # Fix variable name consistency
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        output = subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path])
        name = output.split(b',')[1][:-3].decode()  # Decode output to string and fix byte splitting

        if name in ['unknown_person', 'no_person_found']:
            util.msg_box('Ups', 'Try again')
        else:
            util.msg_box('Welcome back', f'Welcome {name}')
            with open(self.log_path, 'a') as f:
                f.write('{},{}\n'.format(name, datetime.datetime.now()))
        
        os.remove(unknown_img_path)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x800")

        self.entry_text_new_user = util.get_entry_text(self.register_new_user_window)
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', "green", self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)
        self.retry_button_register_new_user_window = util.get_button(self.register_new_user_window, "Retry", "red", self.try_again_register_new_user)
        self.retry_button_register_new_user_window.place(x=750, y=400)

        # Webcam capture label
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=800, height=500)

        self.add_img(self.capture_label)

        self.entry_text_new_user.place(x=750, y=150)
        self.text_label = util.get_text_label(self.register_new_user_window, "Enter username")
        self.text_label.place(x=750, y=100)

    def add_img(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def accept_register_new_user(self):
        name = self.entry_text_new_user.get(1.0, "end-1c")
        cv2.imwrite(os.path.join(self.db_dir, f'{name}.jpg'), self.register_new_user_capture)
        util.msg_box('Success', 'Done')

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()


# import face_recognition
# import pickle
# import os

# # Directory where images are stored
# images_dir = './db'

# # Dictionary to hold face encodings and corresponding names
# face_encodings_dict = {}

# # Iterate through all images in the directory
# for image_file in os.listdir(images_dir):
#     if image_file.endswith('.jpg'):
#         image_path = os.path.join(images_dir, image_file)
#         image = face_recognition.load_image_file(image_path)
#         encodings = face_recognition.face_encodings(image)
        
#         if encodings:
#             # Assuming one face per image, use the first encoding
#             face_encodings_dict[os.path.splitext(image_file)[0]] = encodings[0]

# # Save encodings to a file
# with open('face_encodings.pkl', 'wb') as f:
#     pickle.dump(face_encodings_dict, f)


# import tkinter as tk
# from PIL import Image, ImageTk
# import util
# import cv2
# import os
# import datetime
# import pickle
# import face_recognition

# # Directory where images are stored
# images_dir = './db'

# # Dictionary to hold face encodings and corresponding names
# face_encodings_dict = {}

# # Iterate through all images in the directory
# for image_file in os.listdir(images_dir):
#     if image_file.endswith('.jpg'):
#         image_path = os.path.join(images_dir, image_file)
#         image = face_recognition.load_image_file(image_path)
#         encodings = face_recognition.face_encodings(image)
        
#         if encodings:
#             # Assuming one face per image, use the first encoding
#             face_encodings_dict[os.path.splitext(image_file)[0]] = encodings[0]

# # Save encodings to a file
# with open('face_encodings.pkl', 'wb') as f:
#     pickle.dump(face_encodings_dict, f)

# class App():
#     def __init__(self):
#         self.main_window = tk.Tk()
#         self.main_window.geometry("1200x800")

#         self.login_button_main_window = util.get_button(self.main_window, 'login', "green", self.login)
#         self.login_button_main_window.place(x=750, y=300)
#         self.register_new_user_button_main_window = util.get_button(self.main_window, 'New User', "red", self.register_new_user, fg='black')
#         self.register_new_user_button_main_window.place(x=750, y=400)

#         # Webcam
#         self.webcam_label = util.get_img_label(self.main_window)
#         self.webcam_label.place(x=10, y=0, width=800, height=500)

#         self.add_webcam(self.webcam_label)

#         self.db_dir = './db'
#         if not os.path.exists(self.db_dir):
#             os.mkdir(self.db_dir)

#         self.log_path = './log.txt'

#         # Load face encodings from file
#         self.encodings_path = 'face_encodings.pkl'
#         if os.path.exists(self.encodings_path):
#             with open(self.encodings_path, 'rb') as f:
#                 self.known_face_encodings = pickle.load(f)
#                 self.known_face_names = list(self.known_face_encodings.keys())
#                 self.known_face_encodings = list(self.known_face_encodings.values())
#         else:
#             self.known_face_encodings = []
#             self.known_face_names = []

#     def add_webcam(self, label):
#         if 'cap' not in self.__dict__:
#             self.cap = cv2.VideoCapture(0)
        
#         self.label = label
#         self.process_webcam()

#     def process_webcam(self):
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.most_recent_capture_arr = frame
#             self.most_recent_capture_pil = Image.fromarray(frame)
#             imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)

#             self.label.imgtk = imgtk
#             self.label.configure(image=imgtk)

#         self.main_window.after(20, self.process_webcam)

#     def login(self):
#         unknown_img_path = './.tmp.jpg'
#         cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

#         unknown_image = face_recognition.load_image_file(unknown_img_path)
#         unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#         results = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding)
#         name = "unknown_person"

#         if True in results:
#             first_match_index = results.index(True)
#             name = self.known_face_names[first_match_index]

#         if name in ['unknown_person', 'no_person_found']:
#             util.msg_box('Ups', 'Try again')
#         else:
#             util.msg_box('Welcome back', f'Welcome {name}')
#             with open(self.log_path, 'a') as f:
#                 f.write('{},{}\n'.format(name, datetime.datetime.now()))
        
#         os.remove(unknown_img_path)

#     def register_new_user(self):
#         self.register_new_user_window = tk.Toplevel(self.main_window)
#         self.register_new_user_window.geometry("1200x800")

#         self.entry_text_new_user = util.get_entry_text(self.register_new_user_window)
#         self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', "green", self.accept_register_new_user)
#         self.accept_button_register_new_user_window.place(x=750, y=300)
#         self.retry_button_register_new_user_window = util.get_button(self.register_new_user_window, "Retry", "red", self.try_again_register_new_user)
#         self.retry_button_register_new_user_window.place(x=750, y=400)

#         # Webcam capture label
#         self.capture_label = util.get_img_label(self.register_new_user_window)
#         self.capture_label.place(x=10, y=0, width=800, height=500)

#         self.add_img(self.capture_label)

#         self.entry_text_new_user.place(x=750, y=150)
#         self.text_label = util.get_text_label(self.register_new_user_window, "Enter username")
#         self.text_label.place(x=750, y=100)

#     def add_img(self, label):
#         imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)

#         self.register_new_user_capture = self.most_recent_capture_arr.copy()

#     def try_again_register_new_user(self):
#         self.register_new_user_window.destroy()

#     def accept_register_new_user(self):
#         name = self.entry_text_new_user.get(1.0, "end-1c")
#         face_encoding = face_recognition.face_encodings(self.register_new_user_capture)[0]

#         # Update known encodings and names
#         self.known_face_encodings.append(face_encoding)
#         self.known_face_names.append(name)

#         # Save updated encodings
#         with open('face_encodings.pkl', 'wb') as f:
#             pickle.dump(dict(zip(self.known_face_names, self.known_face_encodings)), f)

#         util.msg_box('Success', 'Done')

#     def start(self):
#         self.main_window.mainloop()

# if __name__ == "__main__":
#     app = App()
#     app.start()
