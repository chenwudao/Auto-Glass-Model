import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('factors2/integrated_eye_data.csv')

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

window_size = 5
df['Pupil_Mean'] = df['Pupil_Diameter_mm'].rolling(window=window_size).mean()

# 删除包含NaN的行
df = df.dropna(subset=['Pupil_Mean', 'Illuminance_Lux', 'Env_Score', 'Cognitive_Load', 'Delta_AL_mm'])

X = df[['Illuminance_Lux', 'Pupil_Mean', 'Env_Score', 'Cognitive_Load']]
y = df['Delta_AL_mm']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df['Age_Group'], random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. 模型初始化与训练
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_scaled, y_train)

# 2. 预测与评估
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f} mm")
print(f"R²: {r2:.4f}")

# 3. 特征重要性分析（SHAP）
import shap

explainer = shap.KernelExplainer(model.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)