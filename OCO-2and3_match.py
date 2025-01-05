import os
from ensurepip import bootstrap
import optuna
import geopandas as gpd
import warnings
from rasterio.mask import mask
import rasterio.features
import joblib
from numpy import zeros, mean, array, datetime64, pi, degrees, arcsin, cos, radians, sin, nan, nanmean
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
import statsmodels.api as sm

warnings.filterwarnings("ignore")

folder1 = fr'D:\Global_SIF_Simulate\global_data\OCO3_grid_global'
folder2 = fr'D:\Global_SIF_Simulate\global_data\OCO2_grid_global_uncorrected_15'

overlap_data = pd.DataFrame()

overlap_threshold = 0.99

for file1 in os.listdir(folder1):
    if file1.endswith('.shp'):
        base_name = os.path.splitext(file1)[0]

        year = 2000 + int(base_name[0:2])
        month = int(base_name[2:4])
        day = int(base_name[4:6])
        hour = int(base_name[6:8])

        file2 = os.path.join(folder2, f"{base_name}.shp")
        if os.path.exists(file2):
            shp1 = gpd.read_file(os.path.join(folder1, file1))
            shp2 = gpd.read_file(file2)

            overlap = gpd.overlay(shp1, shp2, how='intersection')

            for i, row in overlap.iterrows():

                shp1_area = shp1.loc[shp1.geometry.intersects(row.geometry), 'geometry'].area.values[0]
                shp2_area = shp2.loc[shp2.geometry.intersects(row.geometry), 'geometry'].area.values[0]
                overlap_area = row.geometry.area
                overlap_ratio = overlap_area / min(shp1_area, shp2_area)

                if overlap_ratio > overlap_threshold:
                    new_row_df = pd.DataFrame([row])
                    centroid = row.geometry.centroid
                    new_row_df['x'] = centroid.x
                    new_row_df['y'] = centroid.y
                    new_row_df['DOY'] = (datetime64(f'{year}-{month:02d}-{day:02d}') - datetime64(
                        f'{year}-01-01')).astype('timedelta64[D]').astype(int) + 1

                    overlap_data = pd.concat([overlap_data, new_row_df], ignore_index=True)
                    print(file1)
                    # print(new_row_df)

overlap_data.drop('FID_1_1', axis=1, inplace=True)
overlap_data.drop('FID_1_2', axis=1, inplace=True)
overlap_data.drop('geometry', axis=1, inplace=True)

overlap_data.to_csv(fr'overlap_grid_15.csv', index=False)

overlap_data = pd.read_csv(fr'overlap_grid_15.csv')
print(overlap_data)
overlap_data = overlap_data.drop_duplicates()
overlap_data = overlap_data.dropna()
print(overlap_data)


def objective(trial, x, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'bootstrap': trial.suggest_categorical('bootstrap', [True]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'oob_score': trial.suggest_categorical('oob_score', [True, False]),
        'criterion': 'absolute_error',
        'n_jobs': -1,
        'random_state': 42,
    }

    model = RandomForestRegressor(**params)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, x, y, cv=cv, scoring=make_scorer(mean_absolute_error)).mean()

    return score


train = overlap_data.iloc[:, 8:-1]
test = overlap_data.iloc[:, 0]
print(train)
print(test)

# sampler = optuna.samplers.TPESampler(seed=42)
# model = optuna.create_study(direction='minimize', sampler=sampler)
# func = lambda trial: objective(trial, train, test)
# model.optimize(func, n_trials=500)
# trial = model.best_trial

traindata, testdata = train_test_split(overlap_data, test_size=0.3, random_state=42)
x_train = traindata.iloc[:, 8:-1]
y_train = traindata.iloc[:, 0]
x_test = testdata.iloc[:, 8:-1]
y_test = testdata.iloc[:, 0]

# best_model = RandomForestRegressor(random_state=42, n_jobs=-1, criterion='absolute_error', max_depth=trial.params['max_depth'], min_samples_split=trial.params['min_samples_split'],
#                                    min_samples_leaf=trial.params['min_samples_leaf'], bootstrap=trial.params['bootstrap'], n_estimators=trial.params['n_estimators'],
#                                    max_features=trial.params['max_features'], oob_score=trial.params['oob_score'])
# best_model.fit(x_train, y_train)
# joblib.dump(best_model, fr'OCO2_to_OCO3.pkl')
best_model = joblib.load(fr'OCO2_to_OCO3.pkl')

y_pred = best_model.predict(x_test)
y_pred2 = best_model.predict(x_train)

print(x_test)
print('MAE(before correction):', mean_absolute_error(x_test['SIF_2'], y_test))
print('MAE(after ml correction):', mean_absolute_error(y_test, y_pred))
print('MAE(after ml correction):', mean_absolute_error(y_train, y_pred2))

x = array([-10, 10])
plt.xlim(-0.1, 0.8)
plt.ylim(-0.1, 0.8)
plt.plot(x, x, color=(38 / 255, 38 / 255, 38 / 255))
plt.scatter(y_test, x_test['SIF_2'], label='Before Correction, MAE=0.102', color=(41 / 255, 56 / 255, 144 / 255))
plt.scatter(y_test, y_pred, label='After Correction, MAE=0.088', color=(191 / 255, 29 / 255, 45 / 255))
plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
plt.xlabel('OCO-3 SIF (W/m2/μm/sr)', fontsize=14)
plt.ylabel('OCO-2 SIF (W/m2/μm/sr)', fontsize=14)
plt.legend(fontsize=14)
plt.savefig(fr'OCO2_to_OCO3.png', dpi=300)
plt.show()

model = LinearRegression()
model.fit(x_train, y_train)
# X2 = sm.add_constant(x_train)
# est = sm.OLS(y_train, X2)
# est2 = est.fit()
# print(est2.summary())
y_pred = model.predict(x_test)
y_pred2 = model.predict(x_train)
print('MAE(before correction):', mean_absolute_error(y_test, x_test['SIF_2']))
print('MAE(after linear correction):', mean_absolute_error(y_test, y_pred))
print('MAE(after linear correction):', mean_absolute_error(y_train, y_pred2))
plt.xlim(-0.1, 0.8)
plt.ylim(-0.1, 0.8)
plt.plot(x, x, color=(38 / 255, 38 / 255, 38 / 255))
plt.scatter(y_test, x_test['SIF_2'], label='Before Correction, MAE=0.102', color=(41 / 255, 56 / 255, 144 / 255))
plt.scatter(y_test, y_pred, label='After Correction, MAE=0.099', color=(191 / 255, 29 / 255, 45 / 255))
plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
plt.xlabel('OCO-3 SIF (W/m2/μm/sr)', fontsize=14)
plt.ylabel('OCO-2 SIF (W/m2/μm/sr)', fontsize=14)
plt.legend(fontsize=14)
plt.savefig(fr'OCO2_to_OCO3_2.png', dpi=300)
plt.show()
