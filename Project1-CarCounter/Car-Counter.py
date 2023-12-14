from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
cap=cv2.VideoCapture("Videos/cars2.mp4")  #For Video


model=YOLO("Yolo-Weights/yolov8l.pt")

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
mask =cv2.imread("Project1-CarCounter/mask.png")

#Tracking
tracker=Sort(max_age=20 , min_hits=3 , iou_threshold=0.3)

limits = [100,297,1000,297]
TotalCount=[]
while True:
    success, img = cap.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imRegion=cv2.bitwise_and(img,mask)

    results=model(imRegion, stream=True)
    detections=np.empty((0, 5))

    for r in results:
        boxes=r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]

            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2) #Her satır, bir nesnenin sol üst ve sağ alt köşesinin piksel koordinatlarını temsil eder.
            

            w , h = x2 - x1 , y2 - y1
    
            
            #Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 #nesne güven değerini daha yuvarlanmış bir şekilde yazdırarak, sonucun daha anlaşılır ve düzenli olmasına yardımcı olur.
            
            # Class Name
            cls=int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # cvzone.putTextRect(img,f'{int(Id)}', (max(0 , x1), max(35, y1)),
                # scale=2, thickness=3, offset=10)   
                #cvzone.cornerRect(img, (x1, y1, w, h),l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    resultsTracker=tracker.update(detections)
    
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, Id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        
        w , h = x2 - x1 , y2 - y1
    
        cvzone.cornerRect(img, (x1 , y1 , w ,h),l=9, rt=2, colorR=(255,0,255))
        print(f"Rectangle Coordinates: {x1}, {y1}, {w}, {h}")

        cvzone.putTextRect(img,f'{int(Id)}', (max(0 , x1), max(35, y1)),
                    scale=2, thickness=3, offset=10)    

        cx, cy = x1+w//2, y1+h//2

        cv2.circle(img,(cx,cy), 5 , (255,0,255),cv2.FILLED)
    
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[2] + 15:
            if TotalCount.count(Id) == 0:
                TotalCount.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 0), 5)
    
    print(f"Box Coordinates: {x1}, {y1}, {x2}, {y2}")
    print(f"Class: {currentClass}, Confidence: {conf}")

    cvzone.putTextRect(img,f' Count  : {len(TotalCount)}', (50, 50))
    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imRegion)
    
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

