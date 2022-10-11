import cv2
import numpy as np
import face_recognition

img_train = face_recognition.load_image_file("train/001_train.jpg")
img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file("test/001_test.jpeg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

face_loc_train = face_recognition.face_locations(img_train)[0]
encode_img_train = face_recognition.face_encodings(img_train)[0]
cv2.rectangle(img_train, (face_loc_train[3], face_loc_train[0]), (face_loc_train[1], face_loc_train[2]), (255, 0, 0), 2)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_img_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (255, 0, 0), 2)

comparison_result = face_recognition.compare_faces([encode_img_train], encode_img_test)
print(comparison_result)

cv2.imshow("Train", img_train)
cv2.imshow("Test", img_test)
cv2.waitKey(0)