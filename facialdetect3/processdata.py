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

def generate_simulated_data(num_children=100, num_sessions_per_child=5):
    """
    生成模拟的儿童眼部生物测量和面部特征数据。
    每个儿童在不同年龄（通过会话数模拟年龄增长）和不同环境/任务下有多次测量。
    """
    data = []
    
    # 模拟年龄组分布 (4-12岁)
    age_groups = list(range(4, 13)) # 4, 5, ..., 12岁
    
    for child_id in range(1, num_children + 1):
        gender = random.choice(['Male', 'Female'])
        ethnicity = random.choice(['East_Asian', 'Caucasian', 'Other'])
        
        # 模拟基线AL和LT，受年龄和种族影响
        baseline_al = np.random.uniform(20.5, 24.5) + (0.5 if ethnicity == 'East_Asian' else 0)
        baseline_lt = np.random.uniform(3.5, 4.0)
        baseline_se = np.random.uniform(-1.0, 1.0)
        
        # 模拟不同年龄点的数据
        for session_idx in range(num_sessions_per_child):
            # 确保年龄跨度，模拟不同年龄组
            age_years = random.choice(age_groups) + np.random.uniform(-0.5, 0.5) 
            age_years = max(4.0, min(12.9, age_years)) # 限制在4-12岁
            
            # 模拟测量情境
            environment_type = random.choice(['Outdoor', 'Indoor_Study', 'Indoor_Play', 'Lab_Controlled'])
            task_type = random.choice(['Reading', 'Screen_Use', 'Outdoor_Play', 'Resting_View_Distance', 'Cognitive_Task'])
            
            # 模拟外部环境指标
            illuminance_lux = np.random.uniform(100, 50000) # 室内100-1000，户外>5000
            if 'Outdoor' in environment_type:
                illuminance_lux = np.random.uniform(5000, 50000)
            elif 'Indoor_Study' in environment_type:
                illuminance_lux = np.random.uniform(300, 700)
            
            color_temp_k = np.random.uniform(2700, 6500)
            current_reading_distance_cm = np.random.uniform(20, 50) if 'Reading' in task_type else np.nan
            current_screen_distance_cm = np.random.uniform(30, 70) if 'Screen_Use' in task_type else np.nan

            # 疲劳程度 (0-10)
            subjective_fatigue_score = np.random.randint(0, 11)
            # 眨眼频率 (次/分钟), 疲劳/屏幕使用可能降低，不适可能升高
            blink_rate_per_minute = np.random.uniform(10, 30) 
            if 'Screen_Use' in task_type:
                blink_rate_per_minute = np.random.uniform(5, 15)
            if subjective_fatigue_score > 7:
                blink_rate_per_minute = np.random.uniform(20, 40) # 疲劳有时也会导致眨眼过多

            # 瞳孔直径 (mm), 受光照和认知负荷影响
            pupil_diameter_mm = np.random.uniform(3.0, 7.0)
            if illuminance_lux > 1000: # 强光下瞳孔缩小
                pupil_diameter_mm = np.random.uniform(2.5, 4.5)
            if 'Cognitive_Task' in task_type: # 认知负荷下瞳孔扩张
                pupil_diameter_mm = np.random.uniform(4.5, 8.0)

            # 眼睛开合度 (EAR), 眯眼会导致EAR降低
            ear_value = np.random.uniform(0.2, 0.4) # 正常范围
            if subjective_fatigue_score > 7: # 疲劳可能眯眼
                ear_value = np.random.uniform(0.15, 0.25)
            if 'Reading' in task_type and current_reading_distance_cm < 25: # 用眼过近可能眯眼
                ear_value = np.random.uniform(0.18, 0.28)

            # 面部动作单元 (AU) 强度 (0-5, 模拟OpenFace输出)
            # AU7 (Lid Tightener - 眼睑收紧), 眯眼时激活
            au7_intensity = np.random.uniform(0, 1.0)
            if ear_value < 0.25: # EAR低时AU7可能激活
                au7_intensity = np.random.uniform(0.5, 5.0) 

            # AU6 (Cheek Raiser - 颧骨抬高), 与眯眼或愉快情绪相关
            au6_intensity = np.random.uniform(0, 1.0)
            if subjective_fatigue_score > 5: # 疲劳时AU6可能激活
                au6_intensity = np.random.uniform(0.3, 3.0)

            # 假设光照强度、用眼距离、瞳孔直径、疲劳等对AL和LT有即时影响
            # AL受到年龄和基线AL影响，以及即时因素的微调
            current_al_mm = (baseline_al + 
                             (0.01 * age_years) + # 年龄增长趋势
                             (-0.000005 * illuminance_lux) + # 强光可能抑制急性AL增长
                             (0.001 * current_reading_distance_cm if not np.isnan(current_reading_distance_cm) else 0) + # 近距离用眼可能增加
                             (0.05 if pupil_diameter_mm > 6.0 else 0) + # 瞳孔扩张可能与AL增长相关
                             (0.02 * subjective_fatigue_score / 10) + # 疲劳可能导致急性AL增长
                             np.random.normal(0, 0.02)) # 随机噪声

            # LT受到年龄和基线LT影响，以及即时因素的微调
            current_lt_mm = (baseline_lt +
                             (0.005 * age_years) + # 年龄增长趋势
                             (0.05 if current_reading_distance_cm < 25 else 0) + # 近距离用眼晶状体变厚
                             (-0.02 if blink_rate_per_minute < 10 else 0) + # 眨眼少可能影响晶状体
                             np.random.normal(0, 0.01)) # 随机噪声
            
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
                'EAR_Value': ear_value, # Eye Aspect Ratio
                'AU6_Intensity': au6_intensity, # Cheek Raiser
                'AU7_Intensity': au7_intensity, # Lid Tightener
                'Subjective_Fatigue_Score': subjective_fatigue_score,
                'Cognitive_Load_Level': random.choice(['Low', 'Medium', 'High']),
                'Time_Since_Last_Break_Minutes': np.random.uniform(0, 120),
                'Hours_Slept_Last_Night': np.random.uniform(7.0, 10.0),
                'Current_AL_mm': current_al_mm,
                'Current_LT_mm': current_lt_mm,
                # 计算Delta值作为另一个目标
                'Delta_AL_mm_From_Baseline': current_al_mm - baseline_al,
                'Delta_LT_mm_From_Baseline': current_lt_mm - baseline_lt
            })
    
    df = pd.DataFrame(data)
    
    # 填充NaN值，这里选择用中位数填充距离，因为它们是条件性的
    df['Current_Reading_Distance_cm'].fillna(df['Current_Reading_Distance_cm'].median(), inplace=True)
    df['Current_Screen_Distance_cm'].fillna(df['Current_Screen_Distance_cm'].median(), inplace=True)
    
    return df

# 生成数据
df = generate_simulated_data(num_children=200, num_sessions_per_child=10) # 200个儿童，每个儿童10次测量，总计2000个样本

print(f"模拟数据集样本量: {len(df)}")
print("\n数据集前5行:")
print(df.head())
print("\n数据集信息:")
df.info()
print("\n各年龄组样本分布:")
df['Age_Group'] = pd.cut(df['Age_Years'], bins=[0, 6, 9, 13], labels=['4-6_Years', '7-9_Years', '10-12_Years'])
print(df['Age_Group'].value_counts())

# 定义特征和目标变量
# 我们将预测 Current_AL_mm 和 Current_LT_mm
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

# 定义数值型和类别型特征
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 划分训练集和测试集 (按Child_ID划分，确保新儿童的泛化能力)
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

print(f"\n训练集样本量: {len(X_train)}")
print(f"测试集样本量: {len(X_test)}")