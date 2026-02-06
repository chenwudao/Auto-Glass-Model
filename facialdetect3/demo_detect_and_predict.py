import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 1. 加载YOLO模型
model = YOLO('yolov8n.pt')  # 可换为人脸专用模型

# 2. 加载/模拟训练好的回归模型（这里只做简单示例，实际应用时请用训练好的模型）
# 假设我们只用Pupil_Diameter_mm和Illuminance_Lux两个特征做回归
scaler = StandardScaler()
svr = SVR()
# 这里仅做演示，实际应加载训练好的scaler和svr模型

# 3. 读取图片并检测人脸
img_path = 'facialdetect3/cat.jpg'
img = cv2.imread(img_path)
results = model(img)

# 4. 对检测到的人脸区域进行特征提取（这里只模拟瞳孔直径和光照）
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 模拟特征提取
        pupil_diameter = np.random.uniform(3.0, 7.0)  # 假设算法提取
        illuminance = np.random.uniform(100, 20000)   # 假设传感器采集
        # 5. 特征标准化与预测（此处仅演示流程）
        X = np.array([[illuminance, pupil_diameter]])
        # X_scaled = scaler.transform(X)  # 实际应先fit scaler
        # pred_al = svr.predict(X_scaled) # 实际应先fit svr
        # 这里只做演示
        pred_al = 23.5 + 0.001 * pupil_diameter - 0.00001 * illuminance
        # 6. 显示结果
        y1_text = max(y1 - 10, 20)
        cv2.putText(img, f"Pred AL: {pred_al:.2f} mm", (x1, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Pupil: {pupil_diameter:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Lux: {illuminance:.0f}", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 缩放图片（例如最大宽度800像素）
max_width = 800
if img.shape[1] > max_width:
    scale = max_width / img.shape[1]
    img = cv2.resize(img, (max_width, int(img.shape[0] * scale)))

cv2.imshow('YOLO Face Detection & AL Prediction', img)
cv2.waitKey(0)
cv2.destroyAllWindows()