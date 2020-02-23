import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from mtcnn.mtcnn import MTCNN
from face_recognition.face_recognition_cli import image_files_in_folder

# นามสกุลไฟล์ที่รองรับ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
detector = MTCNN()
# param "train_dir" โฟล์เดอร์ที่เก็บรูปภาพไว้ train โมเดล
# param "model_save_path" ที่เก็บไฟล์โมเดลที่ใช้ train

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    # เช็คสกุลไฟล์ที่รับเข้ามา
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    # โหลดโมเดล
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # โหลดภาพจาก path
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img,model='cnn')
    # If no faces are found in the image, return an empty result.
    # เช็คว่ามีใบหย้าในรูปหรือไม่
    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    # print(X_face_locations)
    # print("xxx")
    # print(faces_encodings)
    # print("yyy")
    # Use the KNN model to find the best matches for the test face
    # ใช้โมเดล KNN เพื่อหาใผลลัพธ์ที่ใกล้เคียงที่สุด
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=9)


    # print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(are_matches)
    # Predict classes and remove classifications that aren't within the threshold
    # print(knn_clf.predict(faces_encodings))
    # print(X_face_locations)
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    # แปลงภาพเป็นภาพสี
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        # วาดกล่องรอบใบหน้า
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # ใส่ชื่อลงในรูปใต้ภาพใบหน้า
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    # Remove the drawing library from memory as per the Pillow docs
    del draw
    # Display the resulting image
    pil_image.show()

if __name__ == "__main__":


    # ใช้โมเดลที่ไดเ้จากการเทรนมาทำนายใบหน้าในรูปภาพ
    for image_file in os.listdir("storage/test"):
        full_file_path = os.path.join("storage/test", image_file)
        print("Looking for faces in {}".format(image_file))
        # Find all people in the image using a trained classifier model

        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("storage/test", image_file), predictions)