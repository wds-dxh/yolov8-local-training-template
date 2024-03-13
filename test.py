import time
import cv2
import os
os.environ['YOLO_VERBOSE'] = str(False)#不打印yolov8信息
from ultralytics import YOLO


# 加载YOLOv8模型
model = YOLO('best.pt')

cap = cv2.VideoCapture(1)
start_time = time.time()

#保存视屏，设置编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
# 遍历视频帧
save_flag = False
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()
    # frame = cv2.imread("two.png")
    # if success:
    if frame is not None:
        # 在该帧上运行YOLOv8推理
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        results = model.predict(frame,conf=0.25,imgsz=(640, 480),max_det=3,save=True)
        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 显示带注释的帧

        #如果按下s键保存视频
        if cv2.waitKey(1) & 0xFF == ord("s"):
            save_flag = True
            print("保存视频")
        if save_flag:
            out.write(annotated_frame)
        cv2.imshow("YOLOv8推理", annotated_frame)
        # 显示FPS
        # cv2.waitKey(0)
        start_time = time.time()
        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()