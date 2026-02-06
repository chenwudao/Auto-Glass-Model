import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def simulate_axial_length_data(num_samples=1000):
    """
    生成模拟的儿童眼轴长度日变化数据集。

    Args:
        num_samples (int): 要生成的模拟数据样本数量，默认为 1000。

    Returns:
        pandas.DataFrame: 包含模拟数据的 DataFrame，列包括：
                          'age_group' (年龄组), 'baseline_axial_length_mm' (基线眼轴长度，毫米),
                          'time_of_day' (一天中的时间), 'axial_length_change_um' (眼轴长度变化量，微米),
                          'axial_length_mm' (眼轴长度，毫米).
    """
    data = []  # 用于存储生成的数据
    age_groups = {
        'Infant (0-1 yr)': {'baseline_range': (16.8, 20.0), 'amplitude_range': (30, 60)},  # 婴儿（0-1岁）：基线范围和变化幅度范围
        'Toddler (2-5 yrs)': {'baseline_range': (21.0, 22.0), 'amplitude_range': (30, 50)},  # 幼儿（2-5岁）：基线范围和变化幅度范围
        'Preschool (6-7 yrs)': {'baseline_range': (22.4, 22.6), 'amplitude_range': (25, 45)},  # 学龄前儿童（6-7岁）：基线范围和变化幅度范围
        'School-aged (8-12 yrs)': {'baseline_range': (23.0, 23.7), 'amplitude_range': (25, 45)},  # 学龄儿童（8-12岁）：基线范围和变化幅度范围
        'Teenager (13-15 yrs)': {'baseline_range': (23.9, 24.2), 'amplitude_range': (20, 40)},  # 青少年（13-15岁）：基线范围和变化幅度范围
        'Teenager (16-18 yrs)': {'baseline_range': (24.4, 24.7), 'amplitude_range': (15, 35)},  # 青少年（16-18岁）：基线范围和变化幅度范围
    }

    for _ in range(num_samples):  # 循环生成指定数量的样本
        # 随机选择一个年龄组
        age_group = random.choice(list(age_groups.keys()))
        group_params = age_groups[age_group]  # 获取选定年龄组的参数

        # 在该年龄组的基线范围内生成随机的基线眼轴长度
        baseline_axial_length_mm = random.uniform(group_params['baseline_range'][0], group_params['baseline_range'][1])

        # 在该年龄组的变化幅度范围内生成随机的每日变化幅度
        amplitude_um = random.uniform(group_params['amplitude_range'][0], group_params['amplitude_range'][1])

        # 模拟24小时内每小时的变化
        start_time = datetime(2025, 5, 7, 0, 0, 0)  # 任意起始日期
        for hour in range(24):
            current_time = start_time + timedelta(hours=hour)
            time_of_day = current_time.strftime('%H:%M:%S')  # 格式化时间

            # 使用余弦函数模拟日变化，峰值在中午12点左右（第12小时）
            phase_shift = -np.pi  # 相位偏移，使波峰从周期的开始出现
            amplitude_rad = (2 * np.pi * hour) / 24  # 将小时转换为弧度
            axial_length_change_um = (amplitude_um / 2) * (np.cos(amplitude_rad + phase_shift) + 1)  # 计算当前小时的变化量

            axial_length_mm = baseline_axial_length_mm + (axial_length_change_um / 1000)  # 将微米转换为毫米并加到基线上

            data.append([age_group, baseline_axial_length_mm, time_of_day, axial_length_change_um, axial_length_mm])

    df = pd.DataFrame(data, columns=['age_group', 'baseline_axial_length_mm', 'time_of_day', 'axial_length_change_um', 'axial_length_mm'])  # 从数据创建DataFrame
    return df

# 生成包含5000个样本的数据集（您可以根据需要调整此数字）
simulated_data = simulate_axial_length_data(num_samples=5000)

# 打印生成的数据集的前几行
print(simulated_data.head())

# 您可以将此数据保存到CSV文件中以用于机器学习
simulated_data.to_csv('/Environment/engineer/axiallength/simulated_data.csv', index=False)