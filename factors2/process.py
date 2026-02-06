import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. 数据加载与预处理
df = pd.read_csv('factors2/integrated_eye_data.csv')

# 2. 多模态环境评分计算（基于模糊逻辑）
def calculate_environment_score(row):
    # 模糊规则：光照>5000 Lux且蓝光<400nm时评分高
    if row['Illuminance_Lux'] > 5000 and row['BlueLight_nm'] < 400:
        return 0.9  # 低风险
    elif row['Illuminance_Lux'] < 300:
        return 0.3  # 高风险
    else:
        return 0.6  # 中等风险
df['Env_Score'] = df.apply(calculate_environment_score, axis=1)

# 还原 Age_Group 标签
def recover_age_group(row):
    if row['Age_Group_3~6岁'] > 0:
        return '3~6岁'
    elif row['Age_Group_7~12岁'] > 0:
        return '7~12岁'
    elif row['Age_Group_13~18岁'] > 0:
        return '13~18岁'
    else:
        return '未知'

df['Age_Group'] = df.apply(recover_age_group, axis=1)

# 4. 时序特征提取（滑动窗口统计）
window_size = 5  # 5分钟窗口
df['Pupil_Mean'] = df['Pupil_Diameter_mm'].rolling(window=window_size).mean()
# df['Blink_Var'] = df['Blink_Rate_per_Min'].rolling(window=window_size).var()  # 数据中无此列，注释掉

# 5. 年龄组分层与SMOTE过采样
X = df[['Illuminance_Lux', 'Pupil_Mean', 'Env_Score', 'Cognitive_Load']]
y = df['Delta_AL_mm']

# 分层划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df['Age_Group'], random_state=42
)

# 6. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)