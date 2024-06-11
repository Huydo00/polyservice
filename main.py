from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch

# load video
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.9:554/onvif1"
anh = "tes1.jpg"
cap = cv2.VideoCapture(anh)

# load model
model = YOLO('yolov8n.pt')

results = model(cap, stream=True, classes=[45], conf=0) #40,41,42,43,44,

while True:
    # read frames
    ret, image = cap.read()
    if ret:
        image = cv2.resize(image, (854, 480))
        results = model(image)
        image = results[0].plot()

        # Chuyển đổi ảnh sang định dạng RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #BGR2LAB

        ###################################################



        ###################################################

        DP = results[0].numpy()
        if len(DP) != 0:
            # plot
            # results
            # export data
            for result in results:
                boxes = result[0].boxes.numpy()
                for box in boxes:
                    if box.cls == 45:
                        print("id", box.id)
                        print("class", box.cls)
                        print("xyxy", box.xyxy)
                        print("conf", box.conf)
                        print("\n")
        cv2.waitKey(1000)
        ###################################################

        # visualize
        cv2.imshow('YOLOV8', image)
        cv2.waitKey(1)
        k = cv2.waitKey(10) & 0xFF  # "ESC exit"
        if k == 27:
            break

    else:
        cap = cv2.VideoCapture(anh)
cap.release()
cv2.destroyAllWindows()




# for box in boxes:
#   x_min, y_min, x_max, y_max, class_







# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # # cv2.rectangle
        # # cv2.putText
        # frame = results[0].plot()
        #
        #
        # DP = results[0].numpy()
        # # if len(DP) != 0:
        #     # plot
        #     # results
        #     # export data
        #     # Object face
        # # for result in results:
        # #     boxes = result[0].boxes.numpy()
        # #     for box in boxes:
        # #
        # #         print("id", box.id)
        # #         print("class", box.cls)
        # #         print("xyxy", box.xyxy)
        # #         print("conf", box.conf)
        # #         print("\n")