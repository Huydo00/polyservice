from ultralytics import YOLO
import cv2
import os

# load video
link_camera1 = "rtsp://admin:BNNNRU@192.168.1.9:554/onvif1"
cap = cv2.VideoCapture("./anhtest/11.jpg")


# load yolov8 model
model = YOLO('yolov8n.pt')

# track objects
results = model(cap, stream=True, classes=[40,41,42,43,44], conf=0)

while True:
    # read frames
    ret, frame = cap.read()
    if ret:
        # detect objects
        results = model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.rectangle
        # cv2.putText
        frame = results[0].plot()

        # resize
        frame = cv2.resize(frame, (854, 480))

        #visualize
        cv2.imshow('YOLOV8', frame)

        k = cv2.waitKey(10) & 0xFF  # "ESC exit"
        if k == 27:
            break

        # DP = results[0].numpy()
        # if len(DP) != 0:
        #     # plot
        #     # results
        #     # export data
        #     for result in results:
        #         boxes = result[0].boxes.numpy()
        #         for box in boxes:
        #             print("id", box.id)
        #             print("class", box.cls)
        #             print("xyxy", box.xyxy)
        #             print("conf", box.conf)
        #             print("\n")
    else:
        cap = cv2.VideoCapture("./anhtest/11.jpg")

cap.release()
cv2.destroyAllWindows()