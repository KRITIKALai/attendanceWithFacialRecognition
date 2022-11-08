import os
import cv2
import ids
import names
import numpy as np
from pyzbar.pyzbar import decode
import face_recognition
from datetime import datetime

def encodeImages(images):
    encodedImages = list()
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodedImage = face_recognition.face_encodings(image)[0]
        encodedImages.append(encodedImage)
    return encodedImages

def recordAttendance(name):
    with open('registry.csv', 'r+') as f:
        allRecords = f.readlines()
        current_time = datetime.now().strftime('%H:%M:%S')
        nameList = list()
        in_out_status = list()
        for record in allRecords:
            entry = record.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f'\n{name},Done, Done, {current_time}')

def face_recognizer(encodedImages):
    capture = cv2.VideoCapture(0)
    while True:
        success, imgOriginal = capture.read()
        if success:
            img = cv2.resize(imgOriginal, (0, 0), None, 0.25, 0.25)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            facesInFrame = face_recognition.face_locations(img)
            encodedFacesInFrame = face_recognition.face_encodings(img, facesInFrame)

            for faceInFrame, encodedFaceInFrame in zip(facesInFrame, encodedFacesInFrame):
                faceDistance = face_recognition.face_distance(encodedImages, encodedFaceInFrame)
                matchedIndex = np.argmin(faceDistance)
                employee_name = employee_names[matchedIndex]

                y1,x2,y2,x1 = faceInFrame
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(imgOriginal, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.rectangle(imgOriginal, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(imgOriginal, employee_name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Security Camera", imgOriginal)
        key = cv2.waitKey(2)
        if key==ord('q'):
            break

    return employee_name

def qrChecker():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if success:
            for info in decode(img):
                data = info.data.decode('utf-8')
                if data in ids.valid_ids:
                    message = "Pass"
                    color = (0, 255, 0)
                else:
                    message = "Fail"
                    color = (0, 0, 255)

                pts_image = np.array([info.polygon], np.int32)
                pts_image = pts_image.reshape((-1, 1, 2))
                cv2.polylines(img, [pts_image], True, color, 5)
                pts_text = info.rect
                cv2.putText(img, message, (pts_text[0], pts_text[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            key = cv2.waitKey(2)
            if key == ord('q'):
                break

            cv2.imshow('Frame', img)

        else:
            break

    return message

if __name__ == '__main__':
    path = 'Employees'
    images, employee_names = list(), list()
    employees = os.listdir(path)
    for employee in employees:
        img = cv2.imread(f'{path}/{employee}')
        images.append(img)
        employee_names.append(names.filename_to_realname[os.path.splitext(employee)[0]])

    encodedImages = encodeImages(images)

    print("Please show your face..")
    name = face_recognizer(encodedImages)

    print("Please Show your ID..")
    validity_message = qrChecker()

    if validity_message=='Pass':
        print("Valid ID")
        recordAttendance(name)
    else:
        print("Invalid ID!!!")
        print("Please try again..")