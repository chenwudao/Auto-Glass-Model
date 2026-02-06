import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import random # 用于模拟数据
import cv2 # 用于计算机视觉
import time
from collections import deque # 用于历史数据缓冲区

def generate_simulated_data(num_children=100, num_sessions_per_child=5):
    data = []
    age_groups = list(range(4, 13)) # 4, 5, ..., 12岁
    
    for child_id in range(1, num_children + 1):
        gender = random.choice(['Male', 'Female'])
        ethnicity = random.choice(['East_Asian', 'Caucasian', 'Other'])
        
        baseline_al = np.random.uniform(20.5, 24.5) + (0.5 if ethnicity == 'East_Asian' else 0)
        baseline_lt = np.random.uniform(3.5, 4.0)
        baseline_se = np.random.uniform(-1.0, 1.0)
        
        for session_idx in range(num_sessions_per_child):
            age_years = random.choice(age_groups) + np.random.uniform(-0.5, 0.5) 
            age_years = max(4.0, min(12.9, age_years)) # 限制在4-12岁
            
            environment_type = random.choice(['Outdoor', 'Indoor_Study', 'Indoor_Play', 'Lab_Controlled'])
            task_type = random.choice(['Reading', 'Screen_Use', 'Outdoor_Play', 'Resting_View_Distance', 'Cognitive_Task'])
            
            illuminance_lux = np.random.uniform(100, 50000)
            if 'Outdoor' in environment_type:
                illuminance_lux = np.random.uniform(5000, 50000)
            elif 'Indoor_Study' in environment_type:
                illuminance_lux = np.random.uniform(300, 700)
            
            color_temp_k = np.random.uniform(2700, 6500)
            current_reading_distance_cm = np.random.uniform(20, 50) if 'Reading' in task_type else np.nan
            current_screen_distance_cm = np.random.uniform(30, 70) if 'Screen_Use' in task_type else np.nan

            subjective_fatigue_score = np.random.randint(0, 11)
            blink_rate_per_minute = np.random.uniform(10, 30) 
            if 'Screen_Use' in task_type:
                blink_rate_per_minute = np.random.uniform(5, 15)
            if subjective_fatigue_score > 7:
                blink_rate_per_minute = np.random.uniform(20, 40)

            pupil_diameter_mm = np.random.uniform(3.0, 7.0)
            if illuminance_lux > 1000:
                pupil_diameter_mm = np.random.uniform(2.5, 4.5)
            if 'Cognitive_Task' in task_type:
                pupil_diameter_mm = np.random.uniform(4.5, 8.0)

            ear_value = np.random.uniform(0.2, 0.4)
            if subjective_fatigue_score > 7:
                ear_value = np.random.uniform(0.15, 0.25)
            if 'Reading' in task_type and current_reading_distance_cm < 25:
                ear_value = np.random.uniform(0.18, 0.28)

            au7_intensity = np.random.uniform(0, 1.0)
            if ear_value < 0.25:
                au7_intensity = np.random.uniform(0.5, 5.0) 

            au6_intensity = np.random.uniform(0, 1.0)
            if subjective_fatigue_score > 5:
                au6_intensity = np.random.uniform(0.3, 3.0)

            current_al_mm = (baseline_al + 
                             (0.01 * age_years) + 
                             (-0.000005 * illuminance_lux) + 
                             (0.001 * current_reading_distance_cm if not np.isnan(current_reading_distance_cm) else 0) + 
                             (0.05 if pupil_diameter_mm > 6.0 else 0) + 
                             (0.02 * subjective_fatigue_score / 10) + 
                             np.random.normal(0, 0.02))

            current_lt_mm = (baseline_lt +
                             (0.005 * age_years) +
                             (0.05 if current_reading_distance_cm < 25 else 0) +
                             (-0.02 if blink_rate_per_minute < 10 else 0) +
                             np.random.normal(0, 0.01))
            
            data.append({
                'Child_ID': child_id,
                'Age_Years': age_years,
                'Gender': gender,
                'Ethnicity': ethnicity,
                'Baseline_AL_mm': baseline_al,
                'Baseline_LT_mm': baseline_lt,
                'Baseline_SE_D': baseline_se,
                'Measurement_Session_ID': f"S{session_idx+1}",
                'Environment_Type': environment_type,
                'Task_Type': task_type,
                'Illuminance_Lux': illuminance_lux,
                'Color_Temp_K': color_temp_k,
                'Current_Reading_Distance_cm': current_reading_distance_cm,
                'Current_Screen_Distance_cm': current_screen_distance_cm,
                'Blink_Rate_per_Minute': blink_rate_per_minute,
                'Pupil_Diameter_mm': pupil_diameter_mm,
                'EAR_Value': ear_value,
                'AU6_Intensity': au6_intensity,
                'AU7_Intensity': au7_intensity,
                'Subjective_Fatigue_Score': subjective_fatigue_score,
                'Cognitive_Load_Level': random.choice(['Low', 'Medium', 'High']),
                'Time_Since_Last_Break_Minutes': np.random.uniform(0, 120),
                'Hours_Slept_Last_Night': np.random.uniform(7.0, 10.0),
                'Current_AL_mm': current_al_mm,
                'Current_LT_mm': current_lt_mm,
                'Delta_AL_mm_From_Baseline': current_al_mm - baseline_al,
                'Delta_LT_mm_From_Baseline': current_lt_mm - baseline_lt
            })
    df = pd.DataFrame(data)
    df['Current_Reading_Distance_cm'].fillna(df['Current_Reading_Distance_cm'].median(), inplace=True)
    df['Current_Screen_Distance_cm'].fillna(df['Current_Screen_Distance_cm'].median(), inplace=True)
    return df

df = generate_simulated_data(num_children=200, num_sessions_per_child=10)

target_al = 'Current_AL_mm'
target_lt = 'Current_LT_mm'

features = [
    'Age_Years', 'Gender', 'Ethnicity', 
    'Baseline_AL_mm', 'Baseline_LT_mm', 'Baseline_SE_D',
    'Environment_Type', 'Task_Type', 'Illuminance_Lux', 'Color_Temp_K',
    'Current_Reading_Distance_cm', 'Current_Screen_Distance_cm',
    'Blink_Rate_per_Minute', 'Pupil_Diameter_mm', 'EAR_Value',
    'AU6_Intensity', 'AU7_Intensity', 'Subjective_Fatigue_Score',
    'Cognitive_Load_Level', 'Time_Since_Last_Break_Minutes', 'Hours_Slept_Last_Night'
]

X = df[features]
y_al = df[target_al]
y_lt = df[target_lt]

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

unique_child_ids = df['Child_ID'].unique()
train_child_ids, test_child_ids = train_test_split(unique_child_ids, test_size=0.2, random_state=42)

df_train = df[df['Child_ID'].isin(train_child_ids)]
df_test = df[df['Child_ID'].isin(test_child_ids)]

X_train = df_train[features]
y_train_al = df_train[target_al]
y_train_lt = df_train[target_lt]

X_test = df_test[features]
y_test_al = df_test[target_al]
y_test_lt = df_test[target_lt]

# 构建AL预测模型
pipeline_al = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.05)) # 使用一些合理的值，实际应由GridSearchCV给出
])
pipeline_al.fit(X_train, y_train_al)
best_al_model = pipeline_al # 赋值给best_al_model

# 构建LT预测模型
pipeline_lt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.05)) # 使用一些合理的值，实际应由GridSearchCV给出
])
pipeline_lt.fit(X_train, y_train_lt)
best_lt_model = pipeline_lt # 赋值给best_lt_model

print("SVR模型已训练并准备用于实时预测。")

# 替换为实际的 YOLO/面部关键点/AU识别模型
# 假设这里有一个预训练的YOLO模型和面部关键点/特征提取模型
# from ultralytics import YOLO # 如果您安装了ultralytics
# yolo_model = YOLO('yolov8n.pt') # 假设加载一个YOLOv8模型，需要针对人脸检测进行微调或使用专门的人脸检测模型

def detect_faces_yolo_simulated(frame):
    # 假设每帧检测到一个人脸，边界框固定
    h, w, _ = frame.shape
    x1, y1, x2, y2 = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
    # 模拟YOLO的输出格式: [x1, y1, x2, y2, confidence, class_id]
    mock_detections = [[x1, y1, x2, y2, 0.95, 0]] # class_id=0 for 'face'
    return mock_detections

def extract_face_features_simulated(face_roi, illuminance_lux, current_reading_distance_cm, task_type, fatigue_score):
    # 模拟眨眼频率 (受疲劳和屏幕使用影响)
    blink_rate = np.random.uniform(5, 30)
    if 'Screen' in task_type and fatigue_score > 5:
        blink_rate = np.random.uniform(5, 12)

    # 模拟瞳孔直径 (受光照和认知负荷影响)
    pupil_diameter = np.random.uniform(3.0, 7.0)
    if illuminance_lux > 10000:
        pupil_diameter = np.random.uniform(2.5, 4.0)
    # 这里的 Cognitive_Task 假设是从当前环境/任务推断而来
    if 'Cognitive' in task_type: # 认知负荷下瞳孔扩张
        pupil_diameter = np.random.uniform(4.5, 8.0)
    
    # 模拟EAR (Eye Aspect Ratio), 眯眼会导致EAR降低
    ear_value = np.random.uniform(0.2, 0.4) # 正常范围
    if fatigue_score > 7 or current_reading_distance_cm < 25:
        ear_value = np.random.uniform(0.15, 0.25)

    # 模拟AU强度
    au6_intensity = np.random.uniform(0.0, 2.0)
    au7_intensity = np.random.uniform(0.0, 2.0)
    if ear_value < 0.25: # EAR低时AU7可能激活
        au7_intensity = np.random.uniform(1.0, 5.0) 

    return {
        'Blink_Rate_per_Minute': blink_rate,
        'Pupil_Diameter_mm': pupil_diameter,
        'EAR_Value': ear_value,
        'AU6_Intensity': au6_intensity,
        'AU7_Intensity': au7_intensity,
    }

def get_realtime_env_data_simulated():
    """
    模拟实时获取环境数据。
    在实际中，这会从光照传感器、深度摄像头等设备读取。
    """
    return {
        'Illuminance_Lux': np.random.uniform(100, 20000), # 模拟变化的光照强度
        'Color_Temp_K': np.random.uniform(2700, 6500),
        'Current_Reading_Distance_cm': np.random.uniform(20, 50), # 模拟实时用眼距离
        'Current_Screen_Distance_cm': np.random.uniform(30, 70),
        'Time_Since_Last_Break_Minutes': np.random.uniform(0, 120),
        'Hours_Slept_Last_Night': np.random.uniform(7.0, 10.0),
    }

# 这些是SVR模型需要的输入特征，在实时预测时作为已知上下文
# 这些值通常是针对被监测的儿童，在监测前或会话开始时确定的。
current_child_data_context = {
    'Child_ID': 10001, # 假设的儿童ID
    'Age_Years': 8.2, # 监测时儿童的年龄
    'Gender': 'Female',
    'Ethnicity': 'East_Asian',
    'Baseline_AL_mm': 23.10, # 该儿童的基线AL
    'Baseline_LT_mm': 3.70, # 该儿童的基线LT
    'Baseline_SE_D': -1.25, # 该儿童的基线屈光度
    'Environment_Type': 'Indoor_Study', # 假设当前环境类型
    'Task_Type': 'Reading', # 假设当前任务类型 (实时可以从用户输入或AI判断)
    'Subjective_Fatigue_Score': 4, # 假设当前疲劳评分 (可由用户输入或AI估算)
    'Cognitive_Load_Level': 'Medium', # 假设当前认知负荷 (可由AI估算)
}

cap = cv2.VideoCapture(0) # 0 表示默认摄像头

if not cap.isOpened():
    print("错误: 无法打开摄像头。请确保摄像头已连接且未被其他程序占用。")
    exit()

frame_count = 0
start_time = time.time()
display_interval = 1 # 每1秒更新一次屏幕上的文本，避免闪烁

last_display_time = time.time()

print("\n--- 启动实时预测系统 (按 'q' 键退出) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (可能已到达视频结尾或摄像头断开)。退出...")
        break

    frame_count += 1
    
    # 1. YOLO人脸检测与定位
    # 在实际中，这里会调用yolo_model.predict(frame)
    detected_faces = detect_faces_yolo_simulated(frame) # 模拟YOLO检测

    for detection in detected_faces:
        x1, y1, x2, y2, confidence, class_id = detection # YOLO的输出格式
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if confidence > 0.5: # 仅处理置信度高的人脸
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 在帧上绘制人脸框
            
            # 裁剪人脸区域
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue # 避免空ROI

            # 2. 实时环境数据获取
            env_data = get_realtime_env_data_simulated()

            # 3. 实时面部/眼部特征计算
            # 传入环境数据和儿童上下文，以便模拟函数可以更真实地生成特征
            face_features = extract_face_features_simulated(
                face_roi, 
                env_data['Illuminance_Lux'], 
                env_data['Current_Reading_Distance_cm'],
                current_child_data_context['Task_Type'], # 任务类型和疲劳等也影响特征
                current_child_data_context['Subjective_Fatigue_Score']
            )

            # 4. 整合所有特征到DataFrame
            # 确保特征列的顺序和类型与训练SVR模型时使用的 `features` 列表完全一致
            input_data_dict = {
                **current_child_data_context, # 儿童的上下文信息
                **env_data,                    # 实时环境数据
                **face_features                # 实时面部/眼部行为特征
            }
            
            # 将字典转换为DataFrame的单行，用于SVR的predict方法
            # 必须严格按照训练时的`features`列表的顺序和包含的列
            input_features_df = pd.DataFrame([input_data_dict])[features]

            # 5. 进行预测
            predicted_al = np.nan # 初始化为NaN
            predicted_lt = np.nan

            try:
                # 使用训练好的Pipeline进行预测，它会处理标准化和独热编码
                predicted_al = al_predictor_pipeline.predict(input_features_df)[0]
                predicted_lt = lt_predictor_pipeline.predict(input_features_df)[0]
            except Exception as e:
                print(f"预测失败: {e}")
                
            # 6. 显示结果 (每隔一段时间更新一次显示，减少闪烁)
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                # 在帧上显示预测结果
                cv2.putText(frame, f"Pred AL: {predicted_al:.2f} mm", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Pred LT: {predicted_lt:.2f} mm", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                
                # 显示实时提取的一些特征
                cv2.putText(frame, f"Lux: {env_data['Illuminance_Lux']:.0f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Blink: {face_features['Blink_Rate_per_Minute']:.1f}", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Pupil: {face_features['Pupil_Diameter_mm']:.2f}", (x1, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"EAR: {face_features['EAR_Value']:.2f}", (x1, y2 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                last_display_time = current_time

    cv2.imshow('Real-time Ocular Data Prediction from Face (Simulated)', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()

end_time_total = time.time()
fps = frame_count / (end_time_total - start_time)
print(f"\n总处理帧数: {frame_count}, 平均FPS: {fps:.2f}")