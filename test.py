import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from datetime import date
from datetime import datetime

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
da = date.today()
cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN)
cv2.setMouseCallback('Frame', RGB)

cap = cv2.VideoCapture(0)
model=YOLO('yolov8n.pt')
fps = int(cap.get(cv2.CAP_PROP_FPS))
filename = str(da)+'.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter(filename,fourcc,14,(1020,500))

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
area1=[(198,499),(807,499),(822,440),(198,440)]
area2=[(174,404),(174,435),(822,435),(806,404)]
entry=set()
entry1=set()
exit=set()
exit1=set()
while True:
    ret,frame = cap.read()
    time = datetime.now()
    frame=cv2.resize(frame,(1020,500))
    results=model.track(frame,persist=True)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    for index,row in px.iterrows():
        le = len(row)-1
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        id=int(row[4])
        d=int(row[le])
        c=class_list[d]
        if 'person' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            #if cy1<(cy+5) and cy1>(cy-5):   
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            #cv2.circle(frame,(x1,y2),4,(0,0,0),-1)
            result1 = cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
            if result1>=0:
                entry.add(id)
            if id in entry:
                result2 = cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
                if result2>=0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    #cvzone.putTextRect(frame,f'{id}',(x1,y1),1,2)
                    entry1.add(id)
            
            '''result2 = cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
            if result1>=0:
                exit.add(id)
            if id in exit:
                result1 = cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
                if result2>=0:
                    #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    #cvzone.putTextRect(frame,f'{id}',(x1,y1),1,2)
                    exit1.add(id)'''
    
               
    l = len(entry1)+1402
    #l1 = len(exit1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==-1:
        cvzone.putTextRect(frame, f'COUNTER:0', (50, 60), 2, 2)
    else:
        cvzone.putTextRect(frame, f'COUNTER:{l}', (50, 60), 2, 2)
    #cv2.putText(frame,f'COUNTER:{l}', (50,60), font, 1.5, (255,255,255), 2)
    #cv2.line(frame,(404,327),(610,351),(0, 255, 0),2)
    #cvzone.putTextRect(frame, f'INSIDE COUNTER:{l-l1}', (50, 100), 2, 2)
    cv2.putText(frame,f'{time}',(750,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
    #cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
    #cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1)
    cv2.imshow("Frame", frame)
    out.write(frame)
    with open('counter.txt','a') as file:
        file.write(f"COUNT: {l} at {time}")
    key = cv2.waitKey(30)
    if key ==27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()