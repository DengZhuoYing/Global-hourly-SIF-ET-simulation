import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import scipy
# import scipy.stats as stats
# import shap
import lightgbm as lgb
import optuna
import numpy as np
import warnings
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats

warnings.filterwarnings("ignore")


def objective(trial, x, y, x2, y2):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 90, 110),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 900, 1100),
        # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1),
        # 'max_depth': trial.suggest_int('max_depth', 5, 20),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        # 'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1000),
        # 'max_bin': trial.suggest_int('max_bin', 255, 1000),
        # 'metric': trial.suggest_categorical('metric', ['rmse', 'mae', 'huber', 'fair']),
        # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    modeltrain = lgb.LGBMRegressor(**params)
    modeltrain.fit(x, y)
    y_model = modeltrain.predict(x2, n_jobs=-1)
    score = r2_score(y2, y_model)

    return score


df = pd.read_csv(r"OCO_global_final_SIFobs.csv")
dftrain = df[(df['YEAR'] != 2021) & (df['YEAR'] != 2022)]
dfvalidation = df[(df['YEAR'] == 2021)]
dftest = df[(df['YEAR'] == 2022)]
print(dftrain.shape, dfvalidation.shape, dftest.shape)
driven_features = ['LANDCOVER', 'COSSZA', 'APAR', 'T2M', 'SM', 'VPD', 'LON', 'LAT', 'DOY', 'DEM']
predict_features = 'SIFobs'

x = dftrain[driven_features].values
y = dftrain[predict_features].values
x2 = dfvalidation[driven_features].values
y2 = dfvalidation[predict_features].values

# sampler = optuna.samplers.TPESampler(seed=42)
# model = optuna.create_study(direction='maximize', sampler=sampler)
# func = lambda trial: objective(trial, x, y, x2, y2)
# model.optimize(func, n_trials=100)
# trial = model.best_trial

x_train = dftrain[driven_features].values
y_train = dftrain[predict_features].values
x_test = dftest[driven_features].values
y_test = dftest[predict_features].values

# lgbm = lgb.LGBMRegressor(verbosity=-1, n_jobs=-1, random_state=42, n_estimators=trial.params['n_estimators'], learning_rate=trial.params['learning_rate'],
#                          num_leaves=trial.params['num_leaves'])
# lgbm.fit(x_train, y_train)
# joblib.dump(lgbm, 'OCO_lgbm.pkl')
lgbm = joblib.load(r"OCO_lgbm.pkl")
print(lgbm.get_params())

start = time.time()
y_predicted = lgbm.predict(x_test, n_jobs=-1)
end = time.time()
print(f'Time：{end - start}')
r2 = r2_score(y_test, y_predicted)
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
print(f'R2: {r2}')
print(f'RMSE: {rmse}')
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
print(f'a: {lr.coef_[0][0]}')
print(f'b: {lr.intercept_[0]}')

# explainer = shap.Explainer(lgbm)
# shap_values = explainer(y_train)
# plt.figure()
# shap.summary_plot(shap_values, y_train, feature_names=x.columns)
# plt.show()


r2 = stats.pearsonr(y_test, y_predicted)[0] ** 2
lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
sample_size = 20000
indices = np.random.choice(len(y_test), size=sample_size, replace=False)
sampled_y_test = y_test[indices]
sampled_y_predicted = y_predicted[indices]
lr_y_predict = lr.predict(y_test.reshape(-1, 1))
xy = np.vstack([sampled_y_test, sampled_y_predicted])
z = np.abs(scipy.stats.gaussian_kde(xy)(xy))
plt.scatter(sampled_y_test, sampled_y_predicted, c=z, cmap='viridis', s=20 * (z + 1))
plt.plot(y_test, lr_y_predict, color='red')
plt.plot([0, 10], [0, 10], '--', color='black')
plt.xlabel('Observed SIF (W/m\u00b2/μm/sr)', fontsize=14)
plt.ylabel('Predicted SIF (W/m\u00b2/μm/sr)', fontsize=14)
plt.annotate(f'y = {lr.coef_[0][0]:.2f}x+{lr.intercept_[0]:.2f}', xy=(0.06, 0.93), xycoords='axes fraction',
             fontsize=14)
plt.annotate(f'R\u00b2 = {r2:.2f}', xy=(0.06, 0.86), xycoords='axes fraction', fontsize=14)
plt.annotate(f'RMSE={rmse:.2f}', xy=(0.06, 0.79), xycoords='axes fraction', fontsize=14)
# plt.annotate(f'n=20000', xy=(0.08, 0.78), xycoords='axes fraction', fontsize=14,)
plt.xlim(0, 2.2)
plt.ylim(0, 2.2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(np.arange(0, 2.5, 0.5), fontsize=14)
plt.yticks(np.arange(0, 2.5, 0.5), fontsize=14)
# plt.title('LightGBM Regressor', fontsize=14, fontweight='bold')
# cb = plt.colorbar()
# cb.set_label('Density', fontsize=14)
# cb.ax.tick_params(labelsize=14, width=2, length=5, color='black')
plt.clim(0, 3)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(fr'SIFoco.png', dpi=300)
plt.show()
