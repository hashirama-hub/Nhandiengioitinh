from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
                    

model = load_model('gender_detection.model')


webcam = cv2.VideoCapture(0)
    
classes = ['Nam','Nu']


while webcam.isOpened():

    
    status, frame = webcam.read()

    
    face, confidence = cv.detect_face(frame)


    # Vòng lặp nếu phát hiện khuôn mặt
    for idx, f in enumerate(face):

                #Nhận 4 điểm góc của khuôn mặt 
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # vẽ hình chữ nhật khi phát hiện khuôn mặt
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # cắt vùng khuôn mặt được xác định
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # tiền xử lý của việc phát hiện giới tính
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Áp dụng phân biệt giới tính vào khuôn mặt
        conf = model.predict(face_crop)[0] 

        # Nhận lable với độ chính xác tối đa
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    
    cv2.imshow("Phan Biet Gioi Tinh", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()