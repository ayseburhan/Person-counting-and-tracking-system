from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
cap = cv2.VideoCapture("Videos/people.mp4")  # Video dosyası


model = YOLO("Yolo-Weights/yolov8l.pt")
classNames=["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
            "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
            "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
            "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
            "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
            "teddy bear","hair drier","toothbrush"
            ]
mask =cv2.imread("PeopleCounter/mask.png")

limitsUp = [150, 250, 400, 250]
limitsDown = [800, 700, 1100, 700]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

TotalCountUp = []
TotalCountDown = []

while True:
    success, img = cap.read()
    if not success:
        break

    mask = cv2.imread("PeopleCounter/mask.png")
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imRegion = cv2.bitwise_and(img, mask)

    results = model(imRegion, stream=True)
    
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["person"] and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result.astype(int)
        w, h = x2 - x1, y2 - y1
        print(result)
        cvzone.cornerRect(img, (x1 , y1 , w ,h),l=9, rt=2, colorR=(255,0,255))
        print(f"Rectangle Coordinates: {x1}, {y1}, {w}, {h}")

        cvzone.putTextRect(img,f'{int(Id)}', (max(0 , x1), max(35, y1)),
                    scale=2, thickness=3, offset=10)   
        cx, cy = x1 + w // 2, y1 + h // 2

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[3] + 15:
            if Id not in TotalCountUp:
                TotalCountUp.append(Id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
    
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[3] + 15:
            if Id not in TotalCountDown:
                TotalCountDown.append(Id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cv2.putText(img, str(int(Id)), (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    
    cv2.putText(img, str(len(TotalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(TotalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
