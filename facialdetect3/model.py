import pandas as pd
import numpy as np

# 构建AL预测模型
print("\n--- 训练AL预测模型 ---")
pipeline_al = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR())
])

param_grid_al = {
    'svr__kernel': ['rbf'], # RBF核适用于非线性关系
    'svr__C': [0.1, 1, 10, 100], # 正则化参数
    'svr__gamma': [0.001, 0.01, 0.1, 1, 'scale'], # RBF核的宽度参数
    'svr__epsilon': [0.01, 0.05, 0.1, 0.2] # 误差容忍带
}

grid_search_al = GridSearchCV(pipeline_al, param_grid_al, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search_al.fit(X_train, y_train_al)

print("\nAL预测模型最佳参数:", grid_search_al.best_params_)
print("AL预测模型最佳交叉验证负均方误差:", grid_search_al.best_score_)

# 评估AL模型在测试集上的性能
best_al_model = grid_search_al.best_estimator_
y_pred_al = best_al_model.predict(X_test)

mse_al = mean_squared_error(y_test_al, y_pred_al)
rmse_al = np.sqrt(mse_al)
mae_al = mean_absolute_error(y_test_al, y_pred_al)
r2_al = r2_score(y_test_al, y_pred_al)

print(f"\nAL预测模型在测试集上的性能:")
print(f"  均方误差 (MSE): {mse_al:.4f}")
print(f"  均方根误差 (RMSE): {rmse_al:.4f}")
print(f"  平均绝对误差 (MAE): {mae_al:.4f}")
print(f"  R平方 (R2): {r2_al:.4f}")

# 构建LT预测模型
print("\n--- 训练LT预测模型 ---")
pipeline_lt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR())
])

param_grid_lt = {
    'svr__kernel': ['rbf'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    'svr__epsilon': [0.01, 0.05, 0.1, 0.2]
}

grid_search_lt = GridSearchCV(pipeline_lt, param_grid_lt, cv=5, 
                              scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search_lt.fit(X_train, y_train_lt)

print("\nLT预测模型最佳参数:", grid_search_lt.best_params_)
print("LT预测模型最佳交叉验证负均方误差:", grid_search_lt.best_score_)

# 评估LT模型在测试集上的性能
best_lt_model = grid_search_lt.best_estimator_
y_pred_lt = best_lt_model.predict(X_test)

mse_lt = mean_squared_error(y_test_lt, y_pred_lt)
rmse_lt = np.sqrt(mse_lt)
mae_lt = mean_absolute_error(y_test_lt, y_pred_lt)
r2_lt = r2_score(y_test_lt, y_pred_lt)

print(f"\nLT预测模型在测试集上的性能:")
print(f"  均方误差 (MSE): {mse_lt:.4f}")
print(f"  均方根误差 (RMSE): {rmse_lt:.4f}")
print(f"  平均绝对误差 (MAE): {mae_lt:.4f}")
print(f"  R平方 (R2): {r2_lt:.4f}")

# 可视化预测结果 (AL)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_al, y_pred_al, alpha=0.6)
plt.plot([y_test_al.min(), y_test_al.max()], [y_test_al.min(), y_test_al.max()], 'r--', lw=2)
plt.xlabel("真实AL (mm)")
plt.ylabel("预测AL (mm)")
plt.title("AL预测模型 - 真实值 vs. 预测值")
plt.grid(True)
plt.show()

# 特征重要性分析 (使用Permutation Importance for SVR)
from sklearn.inspection import permutation_importance

# AL模型
print("\n--- AL模型特征重要性 (Permutation Importance) ---")
# 需要原始的X_test和y_test，以及一个能直接预测的model对象
# preprocessor已经集成到pipeline中了，所以可以直接使用best_al_model
results_al = permutation_importance(best_al_model, X_test, y_test_al, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx_al = results_al.importances_mean.argsort()[::-1] # 降序
for i in sorted_idx_al:
    print(f"{X_test.columns[i]}: {results_al.importances_mean[i]:.4f} +/- {results_al.importances_std[i]:.4f}")

# LT模型
print("\n--- LT模型特征重要性 (Permutation Importance) ---")
results_lt = permutation_importance(best_lt_model, X_test, y_test_lt, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx_lt = results_lt.importances_mean.argsort()[::-1] # 降序
for i in sorted_idx_lt:
    print(f"{X_test.columns[i]}: {results_lt.importances_mean[i]:.4f} +/- {results_lt.importances_std[i]:.4f}")