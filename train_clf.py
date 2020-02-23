import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from mtcnn.mtcnn import MTCNN
from face_recognition.face_recognition_cli import image_files_in_folder
from google.cloud import storage
from firebase import firebase

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

firebase = firebase.FirebaseApplication('https://face-recognize-20711.firebaseio.com/')
client = storage.Client()
bucket = client.get_bucket('gs://face-recognize-20711.appspot.com')

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
            # print(image.shape)
            # หาตำแหน่งของใบหน้า
            face_bounding_boxes = face_recognition.face_locations(image)
            # print(class_dir)
            # print(face_bounding_boxes)
            # face_bounding_boxes = detector.detect_faces(image)
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                # เช็คว่าในรูปที่เทรนมีใบหน้าหลายคนหรือไม่ถ้ามีหลายคนจะข้ามภาพนั้นไป
                # print("OK")
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:

                # Add face encoding for current image to the training set
                # ใส่ข้อมูลลงอาเรย์
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                # print(X[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    # กำหนดจำนวนเพื่อนบ้านสำหรับการทำ KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        # print(n_neighbors)
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # สร้างโมเดล และ เทรน
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    # print(knn_clf)
    # เก็บโมเดลที่เทรนมา
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

if __name__ == "__main__":
    # Once the model is trained and saved, you can skip this step next time.
    # เริ่มทำการเทรนโมเดลแล้วเซฟลงไดร์ฟ
    for idX in range(1,len(result)):
        for imageY in range(0,64):
            pathWay = 'Tain' + str(idX) + '.' + str(imageY)
            print(pathWay)
            imageBlob = bucket.blob(pathWay)
            print(imageBlob)
            url = imageBlob.public_url
            req = urllib.urlopen(url)
            arr = np.asarray(bytearray(req.read()),dtype=np.uint8)
            img = cv2.imdecode(arr,-1)
    print("Training KNN classifier...")
    # classifier = train("storage/trained", model_save_path="trained_knn_model.clf") #, n_neighbors=2)
    print("Training complete!")