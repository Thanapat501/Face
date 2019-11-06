from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf
import face_recognition

# initialise the detector class.
# สร้างตัว detect ใบหน้าจาก class MTCNN
detector = MTCNN()

# load an image as an array
# โหลดรูปภาพ
image = face_recognition.load_image_file(r'C:\Users\Jammy\Desktop\Project\dataFace\Aom\face_10_3.jpg')
folder = r'C:\Users\Jammy\Desktop\Project\dataFace\Aom/'


# หาตำแหน่งใบหน้าบนรูปภาพ

face_locations = detector.detect_faces(image)

# draw bounding box and five facial landmarks of detected face
# วาดขอบเขตและสถานที่สำคัญ 5 ตำแหน่งบนใบหน้าของใบหน้าที่ตรวจพบ

for face in zip(face_locations):
    (x, y, w, h) = face[0]['box']
    landmarks = face[0]['keypoints']
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for key, point in landmarks.items():
        cv2.circle(image, point, 2, (255, 0, 0), 6)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
