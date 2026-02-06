import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from aldataop import simulated_data  # 导入生成的 simulated_data

# 设置中文字体为 SimHei（黑体），避免中文乱码
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 1. 准备数据
df = simulated_data.copy()

# 将时间转换为小时的数值表示
df['hour'] = df['time_of_day'].apply(lambda x: int(x.split(':')[0]))

# 使用LabelEncoder编码年龄组
label_encoder = LabelEncoder()
df['age_group_encoded'] = label_encoder.fit_transform(df['age_group'])

# 选择特征和目标变量
features = ['age_group_encoded', 'hour']
target = 'axial_length_mm'

X = df[features]
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练模型 (选择随机森林回归模型)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 3. 创建预测函数
def predict_axial_length(age_group_str, time_of_day_str):
    """
    预测给定年龄组和时间段的眼轴长度。

    Args:
        age_group_str (str): 年龄组字符串 (例如: 'School-aged (8-12 yrs)').
        time_of_day_str (str): 一天中的时间字符串 (例如: '10:30:00').

    Returns:
        float: 预测的眼轴长度 (毫米)，如果输入无效则返回 None。
    """
    try:
        age_group_encoded = label_encoder.transform([age_group_str])[0]
        hour = int(time_of_day_str.split(':')[0])
        prediction = model.predict([[age_group_encoded, hour]])[0]
        return prediction
    except Exception as e:
        print(f"预测出错: {e}")
        return None

# 4. 对一个个体进行一天所有不同时间段的输出并保存图表
def save_individual_daily_axial_length(age_group_str, output_dir):
    """
    模拟并保存一个个体在一天中不同时间段的眼轴长度变化图表。

    Args:
        age_group_str (str): 要模拟的个体的年龄组字符串。
        output_dir (str): 保存图表的文件夹路径。
    """
    hours = range(24)
    predicted_lengths = []
    time_points = []

    for hour in hours:
        time_str = f"{hour:02d}:00:00"
        prediction = predict_axial_length(age_group_str, time_str)
        if prediction is not None:
            predicted_lengths.append(prediction)
            time_points.append(time_str)

    if predicted_lengths:
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, predicted_lengths, marker='o', linestyle='-')
        plt.title(f'{age_group_str} 个体一日内眼轴长度变化模拟')
        plt.xlabel('时间 (小时)')
        plt.ylabel('眼轴长度 (毫米)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表到指定文件夹
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{age_group_str.replace(' ', '_').replace('/', '_')}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"图表已保存到: {output_path}")
    else:
        print(f"无法为年龄组 {age_group_str} 生成可视化数据。")

# 5. 保存所有年龄组的每日变化图表
def save_all_age_groups(output_dir="imgs"):
    """
    保存所有年龄组的每日眼轴长度变化图表。

    Args:
        output_dir (str): 保存图表的文件夹路径。
    """
    age_groups = df['age_group'].unique()
    for age_group in age_groups:
        save_individual_daily_axial_length(age_group, output_dir)

# 6. 用户输入预测
def user_input_prediction():
    """
    允许用户输入年龄组和时间，预测眼轴长度。
    """
    print("可用的年龄组:")
    for age_group in df['age_group'].unique():
        print(f"- {age_group}")
    
    age_group_str = input("请输入年龄组: ")
    time_of_day_str = input("请输入时间 (格式: HH:MM:SS): ")
    
    predicted_length = predict_axial_length(age_group_str, time_of_day_str)
    if predicted_length is not None:
        print(f"预测 {age_group_str} 在 {time_of_day_str} 的眼轴长度为: {predicted_length:.2f} 毫米")
    else:
        print("输入无效，无法进行预测。")

# 示例：保存所有年龄组的图表到 imgs 文件夹
save_all_age_groups(output_dir="imgs")

# 示例：用户输入预测
user_input_prediction()