import pandas as pd

def generate_synthetic_samples_conceptual(X_small, num_synthetic_samples=1000):

    if len(X_small) == 0:
        return pd.DataFrame()

    synthetic_data_list = []
    
    # 随机采样并添加噪声
    for _ in range(num_synthetic_samples // 2):
        idx = np.random.randint(0, len(X_small))
        noisy_sample = X_small.iloc[idx].copy()
        
        # 对数值特征添加少量高斯噪声
        for col in X_small.select_dtypes(include=np.number).columns:
            noise_scale = X_small[col].std() * 0.05 # 5% 的标准差作为噪声尺度
            noisy_sample[col] += np.random.normal(0, noise_scale)
        synthetic_data_list.append(noisy_sample)

    # 简单插值（随机选择两个样本进行线性插值）
    for _ in range(num_synthetic_samples // 2):
        idx1, idx2 = np.random.choice(len(X_small), 2, replace=False)
        sample1 = X_small.iloc[idx1]
        sample2 = X_small.iloc[idx2]
        alpha = np.random.uniform(0, 1) # 插值系数

        interpolated_sample = sample1 * alpha + sample2 * (1 - alpha)
        
        # 对于分类特征，随机选择其中一个样本的值
        for col in X_small.select_dtypes(include='object').columns:
            interpolated_sample[col] = random.choice([sample1[col], sample2[col]])
        
        synthetic_data_list.append(interpolated_sample)

    synthetic_X = pd.DataFrame(synthetic_data_list)
    # 注意：生成合成数据的目标变量 y_synthetic 需要根据 X_synthetic 重新计算或通过模型预测。
    # 最理想的方式是使用生成模型（如GAN）同时生成X和y。
    return synthetic_X
