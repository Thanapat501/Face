import sys
import cv2
import face_recognition
import dlib

from skimage import io

# Take the image file name from the command line
# โหลดรูปภาพจากไฟล์ในเครื่อง
file_name = (r'C:\Users\Jammy\Desktop\Project\dataFace\cher_7.jpg')

# Create a HOG face detector using the built-in dlib class
# สร้างตัว detect ใบหน้าจาก class ของ dlib
face_detector = dlib.get_frontal_face_detector()

# ใช้โชว์รูป
win = dlib.image_window()

# Load the image into an array
# โหลดรูปภาพจากไฟล์ในเครื่อง
image = face_recognition.load_image_file(r'C:\Users\Jammy\Desktop\Project\dataFace\cher_7.jpg')

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
# นำรูปมาทำการ detect ใบหน้า
detected_faces = face_detector(image, 1)

# แสดงผลการ detect ใบหน้าที่สามารถตรวจจับได้ในรูปภาพนั้นๆ
print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
# แสดงรูปภาพ
win.set_image(image)

# Loop through each face we found in the image
# ลูปเพื่อวาดกล่องสี่เหลี่ยมล้อมใบหน้าที่ detect ได้

for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges

    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))
    # Draw a box around each face we found

    win.add_overlay(face_rect)
# Wait until the user hits <enter> to close the window

dlib.hit_enter_to_continue()