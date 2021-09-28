from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from wwo_hist import retrieve_hist_data
from datetime import datetime, timedelta
from seffaflik.elektrik import uretim
from scipy import interpolate
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from xgboost import XGBRegressor
from hyperopt.pyll import scope
import time
import math

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
presentday = datetime.now()
tomorrow = (presentday + timedelta(1)).strftime('%Y-%m-%d')


start = pd.to_datetime('2020-07-01')
end = pd.to_datetime('2021-09-07')
rng = pd.date_range(start, end, freq='H')
len(rng)
rng = rng[0:-1]
len(rng)
rng = pd.DataFrame(rng)
rng = rng.set_axis(['DateTime'], axis=1)

#%% Initialize XGBoostRegression

def XGBoostRegression(X_train,y_train,X_test):

    parameters_for_testing = {
    'colsample_bytree':[0.4],
    'gamma':[0,0.1,0.3],
    'min_child_weight':[10],
    'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth':[3,5],
    'n_estimators':[100],
    'reg_alpha':[1e-5],
    'reg_lambda':[1e-5],
    'subsample':[0.6,0.95]  
    }

                    
    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
                                      min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)
        
    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,verbose=10,scoring='neg_mean_squared_error')
    # gsearch1.best_estimator_.feature_importance()
    gsearch1.fit(X_train,y_train)
    Xgbpred = gsearch1.predict(X_test)

    return Xgbpred

# def XGBoostRegression(X_train,y_train,X_test,parameters_for_testing2):

#     # parameters_for_testing2 = {
#     # 'colsample_bytree':[0.4],
#     # 'gamma':[0,0.1,0.3],
#     # 'min_child_weight':[10],
#     # 'learning_rate':[0.1],
#     # 'max_depth':[3,5],
#     # 'n_estimators':[100],
#     # 'reg_alpha':[1e-5],
#     # 'reg_lambda':[1e-5],
#     # 'subsample':[0.6,0.95]  
#     # }
    
#     xgb_model = XGBRegressor(
#         n_estimators = parameters_for_testing2[0],
#         max_depth = 'gbtree',
#         colsample_bytree = parameters_for_testing2[2],
#         learning_rate = parameters_for_testing2[3],
#         max_depth = int(parameters_for_testing2[4]),
#         n_estimators = int(parameters_for_testing2[5]),
#         objective = 'reg:squarederror',
#         subsample = parameters_for_testing2[7]
#                           )
    
#     xgb_model.fit(X_train, y_train)

                    
#     # xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
#     #                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)
#     # xgb_model = xgboost.XGBRegressor(**parameters_for_testing2)
    
#     # gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing2, n_jobs=-1, verbose=1)
#     xgb_model.fit(X_train,y_train)
#     Xgbpred = xgb_model.predict(X_test)
    
#     return Xgbpred



#%% DATA PROCESSING

#URETIM
Target = uretim.gerceklesen(baslangic_tarihi = '2020-07-01', bitis_tarihi = '2021-09-06', santral_id = "2381")
Target['DateTime'] = pd.to_datetime(Target.Tarih) + Target.Saat.astype('timedelta64[h]')
Target = Target.set_index('DateTime')
Target = pd.DataFrame(Target['Rüzgar'])

frequency=1
start_date = '01-JUL-2020'
end_date = yesterday
api_key = '1aa24f5e16be4fc7bba204316210609'
location_list = ['41.143107,28.317183']
hist_weather_data = retrieve_hist_data(api_key, location_list,
                                        start_date, end_date, 
                                        frequency, location_label = False, 
                                        export_csv = False, store_df = True)
df = hist_weather_data[0]
df['hour'] = pd.DatetimeIndex(df['date_time']).hour
df['month'] = pd.DatetimeIndex(df['date_time']).month
df = df.set_index('date_time')

df = df.drop('maxtempC', axis = 1)
df = df.drop('mintempC', axis = 1)
df = df.drop('sunHour', axis = 1)
df = df.drop('uvIndex', axis = 1)
df = df.drop('moon_illumination', axis = 1)
df = df.drop('moonrise', axis = 1)
df = df.drop('moonset', axis = 1)
df = df.drop('sunrise', axis = 1)
df = df.drop('sunset', axis = 1)
df = df.drop('DewPointC', axis = 1)
df = df.drop('FeelsLikeC', axis = 1)
df = df.drop('HeatIndexC', axis = 1)
df = df.drop('WindChillC', axis = 1)
df = df.drop('visibility', axis = 1)
df = df.drop('location', axis = 1)
df = df.drop('totalSnow_cm', axis = 1)

df = df.astype(float)

df['Sin_windspeedKmph'] = np.sin(df['windspeedKmph'])
df['Cos_windspeedKmph'] = np.cos(df['windspeedKmph'])

df['wind_scalar^2'] = df['windspeedKmph']**2
df['wind_scalar^3'] = df['windspeedKmph']**3
# df['wind_scalar^4'] = df['windspeedKmph']**4


df['DateTime'] = rng

Target = pd.merge(Target, rng, on="DateTime", how="outer")
df = df.set_index('DateTime')

Target = Target.set_index('DateTime')
Target = Target.sort_values(by='DateTime', ascending=True)
Target['Rüzgar'] = Target['Rüzgar'].interpolate(method='pad', limit=1)


#%% TRAINING AND VALIDATION SET
df = df.values
Target = Target.values

# space = {
#     'n_estimators' : scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
#     'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 1)),
#     'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
#     'subsample': hp.uniform('subsample', 0.8, 1.0),
#     'colsample_bytree' : hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
#     'alpha' : hp.choice('alpha', np.arange(0.0, 1.1, 0.1)),
#     'objective':  hp.choice('objective', ['reg:squarederror']),
#     'booster': hp.choice('booster', ['gbtree'])
# }

# def xgb_eval_mae(yhat, dtrain):
#     y = dtrain.get_label()
#     return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


# def objective(space):
#     clf = xgb.XGBRegressor(n_estimators = space['n_estimators'],
#                             max_depth = space['max_depth'],
#                             min_child_weight = space['min_child_weight'],
#                             subsample = space['subsample'],
#                             learning_rate = space['learning_rate'],
#                             gamma = space['gamma'],
#                             colsample_bytree = space['colsample_bytree'],
#                             objective='reg:linear'
#                             )

    
#     evaluation = [( X_train, y_train), ( X_test, y_test)]
    
#     clf.fit(X_train, y_train, eval_set=evaluation, eval_metric = 'mae')

#     pred = clf.predict(X_test)
#     mae = mean_absolute_error((y_test), (pred))
#     return{'loss':mae, 'status': STATUS_OK }


# k = 0
# n_splits=5

# tscv = TimeSeriesSplit(max_train_size=None, n_splits=n_splits)
# RMSE = np.zeros(n_splits)
# MAE = np.zeros(n_splits)
# print(tscv)
# for train_index, test_index in tscv.split(df):
#     print("TRAIN:", train_index, "TEST:", test_index)    
#     X_train, X_test = df[train_index], df[test_index]
#     y_train, y_test = Target[train_index], Target[test_index]
#     trials = Trials()
#     parameters_for_testing = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=30, # changetrials=trials)
#     print(parameters_for_testing)
#     Xgbpred = XGBoostRegression(X_train,y_train,X_test,parameters_for_testing)
#     RMSE[k] = rmse(Xgbpred, y_test)
#     MAE[k] = mean_absolute_error(y_test, Xgbpred)
#     x_ax = range(len(y_test))
#     plt.plot(x_ax, list(y_test), linewidth=1.5, label="original")
#     plt.plot(x_ax, list(Xgbpred), linewidth=1.5, label="predicted")
#     plt.title("y-test and y-predicted data")
#     plt.xlabel('Hours')
#     plt.ylabel('MWh')
#     plt.legend(loc='best',fancybox=True, shadow=True)
#     plt.grid(True)
#     plt.show()
#     k = k + 1
#     # if k == 6:
#     #     break
    
    
k = 0
n_splits=10

tscv = TimeSeriesSplit(max_train_size=None, n_splits=n_splits)
RMSE = np.zeros(n_splits)
MAE = np.zeros(n_splits)
print(tscv)
for train_index, test_index in tscv.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)    
    X_train, X_test = df[train_index], df[test_index]
    y_train, y_test = Target[train_index], Target[test_index]    
    Xgbpred = XGBoostRegression(X_train,y_train,X_test)
    RMSE[k] = rmse(Xgbpred, y_test)
    MAE[k] = mean_absolute_error(y_test, Xgbpred)
    x_ax = range(len(y_test))
    plt.plot(x_ax, list(y_test), linewidth=1.5, label="original")
    plt.plot(x_ax, list(Xgbpred), linewidth=1.5, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('TL/MWhr')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    k = k + 1
    if k == 6:
        break
    
#%% TEST SET

frequency=1
start_date = today
end_date = tomorrow
api_key = '1aa24f5e16be4fc7bba204316210609'
location_list = ['41.1431070,28.3171830']

hist_weather_data2 = retrieve_hist_data(api_key, location_list,
                                        start_date, end_date, 
                                        frequency, location_label = False, 
                                        export_csv = False, store_df = True)

X_test2 = hist_weather_data2[0]
X_test2 = X_test2[24:]
X_test2['hour'] = pd.DatetimeIndex(X_test2['date_time']).hour
X_test2['month'] = pd.DatetimeIndex(X_test2['date_time']).month
X_test2 = X_test2.set_index('date_time')

X_test2 = X_test2.drop('maxtempC', axis = 1)
X_test2 = X_test2.drop('mintempC', axis = 1)
X_test2 = X_test2.drop('sunHour', axis = 1)
X_test2 = X_test2.drop('uvIndex', axis = 1)
X_test2 = X_test2.drop('moon_illumination', axis = 1)
X_test2 = X_test2.drop('moonrise', axis = 1)
X_test2 = X_test2.drop('moonset', axis = 1)
X_test2 = X_test2.drop('sunrise', axis = 1)
X_test2 = X_test2.drop('sunset', axis = 1)
X_test2 = X_test2.drop('DewPointC', axis = 1)
X_test2 = X_test2.drop('FeelsLikeC', axis = 1)
X_test2 = X_test2.drop('HeatIndexC', axis = 1)
X_test2 = X_test2.drop('WindChillC', axis = 1)
X_test2 = X_test2.drop('visibility', axis = 1)
X_test2 = X_test2.drop('location', axis = 1)
X_test2 = X_test2.drop('totalSnow_cm', axis = 1)


X_test2 = X_test2.astype(float)

X_test2['Sin_windspeedKmph'] = np.sin(X_test2['windspeedKmph'])
X_test2['Cos_windspeedKmph'] = np.cos(X_test2['windspeedKmph'])

X_test2['wind_scalar^2'] = X_test2['windspeedKmph']**2
X_test2['wind_scalar^3'] = X_test2['windspeedKmph']**3
# X_test2['wind_scalar^4'] = X_test2['windspeedKmph']**4

X_test2 = X_test2.values

Xgbpred2 = XGBoostRegression(X_train,y_train,X_test2)

x = range(len(Xgbpred2))
ax = plt.figure().add_subplot(111)
ax.plot(x, list(Xgbpred2), linewidth=1.5)
ax.set_ylim(0, 6)
ax.set_title("{} GAZİ".format(end_date), fontsize=10)
ax.set_xlabel('Hours', fontsize=12)
ax.set_ylabel('MWh', fontsize=12)
ax.legend(loc="best", prop={'size': 8})
ax.grid(True)

# Xgbpred2 = round_up(Xgbpred2, decimals=1)
Xgbpred2 = pd.DataFrame(Xgbpred2)
Xgbpred2.to_excel('GAZİ.xlsx')




