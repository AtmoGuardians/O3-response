"""
@author: Yang Siyu
"""
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

start_time = time.time()  # 获取开始时间

##################
### 准备数据集 ###
##################
feature = pd.read_csv('/home/yangsy/script/paper_ozone_trend/RF/RF+SHAP/dataset/FWP_rural_2015-2019.csv')

feature_columns = ["anox", "snox", "avoc", "bvoc", "rain", "t2", "u10", "v10",
           "rh", "sp", "radiation", "no2_obs", "no2", "hcho", "lat", "lon", "year", "doy"]
 
x = feature[feature_columns]     # 筛选后作为特征变量
y = feature["ozone"]             # 目标变量
print('x: ')
print(x)
print('y: ')
print(y)   
     
# 构建训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = XGBRegressor()

# 设置参数
# 构建超参数网格
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': np.arange(200, 800, 100),  
    'random_state': [42],
    'max_depth': range(2, 8),
    'reg_alpha': [0.01, 0.1],
    'reg_lambda': [0.01, 0.1]
    }
# GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=24, verbose=2)

# 在训练集上执行网格搜索
grid_search.fit(x_train, y_train)

# 获取所有组合的结果
results = grid_search.cv_results_

# 将结果转换为 DataFrame 以便于查看
results_df = pd.DataFrame(results)

# 输出每个组合的性能指标
for i in range(len(results_df)):
    print(f"组合 {i + 1}:")
    print(f"学习率: {results_df['param_learning_rate'][i]}")
    print(f"n_estimators: {results_df['param_n_estimators'][i]}")
    print(f"max_depth: {results_df['param_max_depth'][i]}")
    print(f"reg_alpha: {results_df['param_reg_alpha'][i]}")
    print(f"reg_lambda: {results_df['param_reg_lambda'][i]}")
    print(f"交叉验证平均 R²: {results_df['mean_test_score'][i]:.4f}")
    print(f"标准差: {results_df['std_test_score'][i]:.4f}")
    print("-" * 30)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score (negative MSE): ", grid_search.best_score_)

xgb = grid_search.best_estimator_
dump(xgb, "/home/yangsy/script/paper_ozone_trend/XGBoost/model/feature18_NO2_obs/xgb_FWP_r_ozone.joblib")
test_score = xgb.score(x_test, y_test)

# 预测结果
ytrpred = xgb.predict(x_train)
ytpred = xgb.predict(x_test)
# 性能验证
print('Train dataset:')
print('R2 of prediction on FWP rural dataset:', r2_score(y_train, ytrpred))
print('RMSE of prediction on FWP rural dataset:', mean_squared_error(y_train, ytrpred, squared=False))
print('MAE of prediction on FWP rural dataset:', mean_absolute_error(y_train, ytrpred))
print('Test dataset:')
print('R2 of prediction on FWP rural dataset:', r2_score(y_test, ytpred))
print('RMSE of prediction on FWP rural dataset:', mean_squared_error(y_test, ytpred, squared=False))
print('MAE of prediction on FWP rural dataset:', mean_absolute_error(y_test, ytpred))

# 计算特征重要性
feat_labels = x.columns
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1] # 下标排序
for f in range(x_train.shape[1]):   
    print("%2d) %-*s %f" % \
          (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

end_time = time.time()  # 获取结束时间
# 计算并打印运行时间
running_time = (end_time - start_time) / 60
print("运行时间：", running_time, " min")
