import cv2
import os
import threading
import numpy as np
import subprocess
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture objects for RGB and IR cameras
rgb_capture = cv2.VideoCapture('/dev/video2')  # RGB camera
#ir_capture = cv2.VideoCapture('/dev/video2')   # IR camera

if not (rgb_capture.isOpened()): # and ir_capture.isOpened()):
    print("Error: One or both cameras not found.")
    exit()

# Create directories to save images
os.makedirs("rgb_faces", exist_ok=True)
os.makedirs("ir_faces", exist_ok=True)

# Counter for generating unique filenames
frame_counter = 0

# Function to calculate image sharpness using the Laplacian method
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Function to capture frames from both cameras and detect faces
def capture_and_detect_faces():
    global frame_counter
    global rgb_capture
    while True:
        ret_rgb, rgb_frame = rgb_capture.read()
        cv2.imshow("rgb_camera", rgb_frame)
        rgb_frame2 = rgb_frame.copy()
        #ret_ir, ir_frame = ir_capture.read()

        if not (ret_rgb): #and ret_ir):
            print("Error: Unable to capture frames.")
            break

        gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        faces_rgb = face_cascade.detectMultiScale(gray_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        #gray_ir = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        #faces_ir = face_cascade.detectMultiScale(gray_ir, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Check if a single full face is detected
        full_face_detected = False
        num_faces = len(faces_rgb)  # Number of faces 
        if num_faces == 1:  # if there is exactly one face
            x, y, w, h = faces_rgb[0]
         #for (x, y, w, h) in faces_rgb:

            if w > 100 and h > 100:  # Adjust these values as needed
                center_x = x + w // 2
                frame_width = rgb_frame.shape[1]
                if frame_width // 3 < center_x < 2 * frame_width // 3:
                  full_face_detected = True

                  face_roi = rgb_frame[y:y + h, x:x + w] #crop the detected face
                  sharpness = calculate_sharpness(face_roi) #sharpness
                  sharpness_threshold = 1000 

                  if sharpness > sharpness_threshold:
                     cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
                     cv2.imshow("Face", face_roi)

                     

                     cv2.imwrite(f'rgb_faces/rgb_frame_{frame_counter}.jpg', face_roi)
                     print(f"Sharpness: {sharpness}. The face comes to the middle of the frame and is sharp enough to save.")
                     subprocess.check_call(['./demo.sh', str(frame_counter)])
                     full_face_detected = True
                     #cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                     #cv2.rectangle(ir_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if num_faces > 1:  # Multiple faces detected
            print("Multiple faces")
            cv2.putText(rgb_frame, "Multiple faces ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ir_frame, "Multiple faces ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for (x, y, w, h) in faces_rgb:
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #for (x, y, w, h) in faces_ir:
            #cv2.rectangle(ir_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow("rgb_camera", rgb_frame)
        #cv2.imshow("ir_camera", ir_frame)  

        # If a full face is detected in middle of the frame, save the frames
        if full_face_detected:
            cv2.imwrite(f'rgb_faces/rgb_frame_{frame_counter}.jpg', rgb_frame2)
            print("The face comes to the middle of the frame.")
            rgb_capture.release()
            ir_capture = cv2.VideoCapture('/dev/video2')
            ret_ir, ir_frame = ir_capture.read()

            cv2.imwrite(f'ir_faces/ir_frame_{frame_counter}.jpg', ir_frame)
            ir_capture.release()
            rgb_capture = cv2.VideoCapture('/dev/video0')
            frame_counter += 1
            time.sleep(3)

            # Call the shell script here with the image filenames
            #subprocess.run(["./demo.sh {frame_counter}"])
            #subprocess.check_call(['./demo.sh', str(frame_counter)])
            print("Subprocess command: ./demo.sh")
            #Sbreak

        #cv2.imshow("rgb_camera", rgb_frame)
        #cv2.imshow("ir_camera", ir_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rgb_capture.release()
    #ir_capture.release()
    cv2.destroyAllWindows()

# Create a thread for capturing and detecting faces
thread_capture = threading.Thread(target=capture_and_detect_faces)

# Start the thread
thread_capture.start()

# Wait for the thread to finish
thread_capture.join()

# Release video capture objects and close OpenCV windows
rgb_capture.release()
#ir_capture.release()

