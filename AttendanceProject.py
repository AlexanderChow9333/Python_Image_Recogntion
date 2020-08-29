import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
myList.pop(0)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        timeList = []
        now = datetime.now()
        now = now.replace(microsecond=0)
        strNow = str(now)
        for line in myDataList:
            entry = line.split(',')
            if name == entry[0]:
                nameList.append(entry[0])
                timeList.append(entry[1].replace('\n',''))
        if name not in nameList:
            f.writelines(f'\n{name},{now}')
            if name == "UNKNOWN":
                os.system("say 'Unknown Person Detected'")
            elif name == "ALEXANDER":
                os.system("say 'Alexander detected'")
            elif name == "MOM":
                os.system("say 'Mom detected'")
            elif name == "DAD":
                os.system("say 'Dad detected'")
            elif name == "ARTHUR":
                os.system("say 'Arthur detected'")
        if name in nameList:
            then = datetime.strptime(timeList[-1],"%Y-%m-%d %H:%M:%S")
            timeDelta = now - then
            timeDelta = timeDelta.total_seconds()/60
            if name == "UNKNOWN":
                os.system("say 'Unknown Person Detected'")
            if timeDelta > 5:
                f.writelines(f'\n{name},{strNow}')
                print("New Entry: " + name)
                if name == "ALEXANDER":
                    os.system("say 'Alexander detected'")
                elif name == "MOM":
                    os.system("say 'Mom detected'")
                elif name == "DAD":
                    os.system("say 'Dad detected'")
                elif name == "ARTHUR":
                    os.system("say 'Arthur detected'")


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.60:
            name = str(classNames[matchIndex].upper())
        else:
            name = 'UNKNOWN'
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, str(100-round(faceDis[matchIndex]*100))+"%", (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255),2)
#
# faceLocTest = face_recognition.face_locations( imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]), (255,0,255),2)
#
#
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)