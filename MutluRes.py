from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
# from WorldWeatherPy import DetermineListOfAttributes
# from WorldWeatherPy import HistoricalLocationWeather
from wwo_hist import retrieve_hist_data
import pandas as pd
import xgboost
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
from seffaflik.elektrik import uretim
from sklearn.model_selection import RandomizedSearchCV

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
presentday = datetime.now()
tomorrow = (presentday + timedelta(1)).strftime('%Y-%m-%d')


start = pd.to_datetime('2020-10-18')
end = pd.to_datetime('2021-09-07')
rng = pd.date_range(start, end, freq='H')
len(rng)
rng = rng[0:-1]
len(rng)
rng = pd.DataFrame(rng)
rng = rng.set_axis(['DateTime'], axis=1)

#%% Initialize XGBoostRegression

def XGBoostRegression(X_train,y_train,X_test):
    # params = { 'max_depth': [3, 5, 6, 10, 15, 20],
    #        'learning_rate': [0.01, 0.1, 0.2, 0.3],
    #        'subsample': np.arange(0.5, 1.0, 0.1),
    #        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
    #        'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    #        'n_estimators': [100, 500, 1000]}
    
    # xgbr = xgboost.XGBRegressor(seed = 20)
    
    # gsearch1 = RandomizedSearchCV(estimator=xgbr,
    #                      param_distributions=params,
    #                      scoring='neg_mean_squared_error',
    #                      n_iter=25,
    #                      verbose=1)

    parameters_for_testing = {
    'colsample_bytree':[0.4],
    'gamma':[0,0.1,0.3],
    'min_child_weight':[10],
    'learning_rate':[0.1],
    'max_depth':[3,5],
    'n_estimators':[200],
    'reg_alpha':[1e-5],
    'reg_lambda':[1e-5],
    'subsample':[0.6,0.95]  
    }
    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
                                      min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, 
                                      scale_pos_weight=1, seed=27)
    
    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing)
    gsearch1.fit(X_train,y_train)
    Xgbpred = gsearch1.predict(X_test)

    return Xgbpred

#%% IMPORT EPIAS PRODUCTION
Target = uretim.gerceklesen(baslangic_tarihi = '2020-10-18', bitis_tarihi = '2021-09-06', santral_id = "2570")
Target['DateTime'] = pd.to_datetime(Target.Tarih) + Target.Saat.astype('timedelta64[h]')
Target = Target.set_index('DateTime')
Target = pd.DataFrame(Target['R??zgar'])

#%% IMPORT WORLDWEATHER
frequency=1
start_date = '18-OCT-2020'
end_date = '06-SEP-2021'
api_key = '1aa24f5e16be4fc7bba204316210609'
location_list = ['38.22,32.27']
hist_weather_data = retrieve_hist_data(api_key,location_list,start_date,end_date,frequency,location_label = False,
                                       export_csv = False, store_df = True)
df = hist_weather_data[0]
df['hour'] = pd.DatetimeIndex(df['date_time']).hour
df['month'] = pd.DatetimeIndex(df['date_time']).month
df = df.rename(columns={'date_time': 'DateTime'})
df['DateTime'] = rng
df = df.set_index('DateTime')

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

Target = pd.merge(Target, rng, on="DateTime", how="outer")
Target = Target.set_index('DateTime')

#%% TRAINING AND VALIDATION SET
df = df.values
Target = Target.values

k = 0
# n_splits=10
for n_splits in range(8,9):
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
        plt.xlabel('Hours')
        plt.ylabel('MWh')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()
        k = k + 1
        if k == 1:
            break

#%% TEST SET

frequency=1
start_date = today
end_date = tomorrow
api_key = '1aa24f5e16be4fc7bba204316210609'
location_list = ['38.22,32.27']
hist_weather_data2 = retrieve_hist_data(api_key, location_list, start_date, end_date, frequency, location_label = False, 
                                        export_csv = False, store_df = True)
X_test2 = hist_weather_data2[0]
X_test2 = X_test2[24:]
X_test2['hour'] = pd.DatetimeIndex(X_test2['date_time']).hour
X_test2['month'] = pd.DatetimeIndex(X_test2['date_time']).month
X_test2 = X_test2.rename(columns={'date_time': 'DateTime'})
X_test2 = X_test2.set_index('DateTime')

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

X_test2 = X_test2.values

Xgbpred = XGBoostRegression(X_train,y_train,X_test2)

x = range(len(Xgbpred))
ax = plt.figure().add_subplot(111)
ax.plot(x, list(Xgbpred), linewidth=1.5)
ax.set_ylim(0, 50)
ax.set_title("{} MUTLU 5".format(end_date), fontsize=10)
ax.set_xlabel('Hours', fontsize=12)
ax.set_ylabel('MWh', fontsize=12)
ax.legend(loc="best", prop={'size': 8})
ax.grid(True)

Xgbpred2 = Xgbpred.copy()
Xgbpred2 = pd.DataFrame(Xgbpred2)
Xgbpred2.to_excel('MUTLU.xlsx')
