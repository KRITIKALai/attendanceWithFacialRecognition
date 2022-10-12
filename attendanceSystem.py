import os
import cv2
import names
import numpy as np
import face_recognition

def encodeImages(images):
    encodedImages = list()
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodedImage = face_recognition.face_encodings(image)[0]
        encodedImages.append(encodedImage)
    return encodedImages

if __name__ == '__main__':
    path = 'Employees'
    images, employee_names = list(), list()
    employees = os.listdir(path)
    for employee in employees:
        img = cv2.imread(f'{path}/{employee}')
        images.append(img)
        employee_names.append(names.filename_to_realname[os.path.splitext(employee)[0]])

    encodedImages = encodeImages(images)

    capture = cv2.VideoCapture(0)
    while True:
        success, img = capture.read()
        if success:
            img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            facesInFrame = face_recognition.face_locations(img)
            encodedFacesInFrame = face_recognition.face_encodings(img, facesInFrame)

            for faceInFrame, encodedFaceInFrame in zip(facesInFrame, encodedFacesInFrame):
                faceDistance = face_recognition.face_distance(encodedImages, encodedFaceInFrame)
                matchedIndex = np.argmin(faceDistance)
                print(employee_names[matchedIndex])

        cv2.imshow("Security Camera", img)
        key = cv2.waitKey(2)
        if key==ord('q'):
            break
