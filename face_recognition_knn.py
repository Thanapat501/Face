import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

# นามสกุลไฟล์ที่รองรับ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# param "train_dir" โฟล์เดอร์ที่เก็บรูปภาพไว้ train โมเดล
# param "model_save_path" ที่เก็บไฟล์โมเดลที่ใช้ train
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    # ลูปวนเช็ครูปในแต่ละบุคคล
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        # ลูปวนรูปภาพในโฟลเดอร์บุคคลนั้นๆ
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # โหลดรูปภาพจาก path ที่ตั้งไว้
            image = face_recognition.load_image_file(img_path)

            # หาตำแหน่งของใบหน้า
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                # เช็คว่าในรูปที่เทรนมีใบหน้าหลายคนหรือไม่ถ้ามีหลายคนจะข้ามภาพนั้นไป
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                # ใส่ข้อมูลลงอาเรย์
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    # กำหนดจำนวนเพื่อนบ้านสำหรับการทำ KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # สร้างโมเดล และ เทรน
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # เก็บโมเดลที่เทรนมา
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

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
    X_face_locations = face_recognition.face_locations(X_img)
    # If no faces are found in the image, return an empty result.
    # เช็คว่ามีใบหย้าในรูปหรือไม่
    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    # ใช้โมเดล KNN เพื่อหาใผลลัพธ์ที่ใกล้เคียงที่สุด
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # Predict classes and remove classifications that aren't within the threshold
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
    # Once the model is trained and saved, you can skip this step next time.
    # เริ่มทำการเทรนโมเดลแล้วเซฟลงไดร์ฟ
    print("Training KNN classifier...")
    classifier = train("storage/trained", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

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