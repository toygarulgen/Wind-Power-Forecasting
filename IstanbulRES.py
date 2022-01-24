from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
# from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import TimeSeriesSplit
from wwo_hist import retrieve_hist_data
from datetime import datetime, timedelta
from seffaflik.elektrik import uretim


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
presentday = datetime.now()
tomorrow = (presentday + timedelta(1)).strftime('%Y-%m-%d')


start = pd.to_datetime('2021-01-29')
end = pd.to_datetime('2021-12-01')
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

                    
    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, 
                                     max_depth=5, min_child_weight=1, gamma=0, 
                                     subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)
    
    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, 
                            n_jobs=6,verbose=10,scoring='neg_mean_squared_error')
    gsearch1.fit(X_train,y_train)
    print (gsearch1.best_params_)
    print('best score')
    print (gsearch1.best_score_)
    Xgbpred = gsearch1.predict(X_test)

    return Xgbpred

#%% Initialize plot
def train_test_plot(a,b):
    x_ax = range(len(a))
    plt.plot(x_ax, list(a), linewidth=1.5, label="original")
    plt.plot(x_ax, list(b), linewidth=1.5, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('Hours')
    plt.ylabel('MWh')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    
#%% DATA PROCESSING
installedcapacity = 181.8

#URETIM
Target = pd.read_excel('IstanbulREStraindata.xlsx')
Target["DateTime"] = Target["DateTime"].astype("datetime64[ns]")
Target = Target.set_index('DateTime')

#%% Refreshing installed capacity and importing weather data
col_names = (Target.index >= "2021-01-29 00:00:00") & (Target.index <= "2021-07-08 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/4.55
col_names = (Target.index >= "2021-07-09 00:00:00") & (Target.index <= "2021-09-17 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/22.55
col_names = (Target.index >= "2021-09-18 00:00:00") & (Target.index <= "2021-10-01 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/49.85
col_names = (Target.index >= "2021-10-02 00:00:00") & (Target.index <= "2021-11-11 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/63.5
col_names = (Target.index >= "2021-11-12 00:00:00") & (Target.index <= "2021-12-02 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/109
col_names = (Target.index >= "2021-12-03 00:00:00") & (Target.index <= "2021-12-18 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/131.75
col_names = (Target.index >= "2021-12-19 00:00:00") & (Target.index <= "2021-12-29 23:00:00")
Target[col_names] = Target[col_names] * installedcapacity/149.95

hist_weather_data = retrieve_hist_data('beb3f237e0274253b3764922210712', ['41.2627,28.1413'],
                                        '29-JAN-2021', '30-NOV-2021', 
                                        1, location_label = False, 
                                        export_csv = False, store_df = True)
df = hist_weather_data[0]
df['hour'] = pd.DatetimeIndex(df['date_time']).hour
df['month'] = pd.DatetimeIndex(df['date_time']).month
df['dayofweek'] = pd.DatetimeIndex(df['date_time']).dayofweek

df = df.set_index('date_time')

#%% Feature Elimination
eliminationfeature=["maxtempC","mintempC","sunHour","uvIndex",
                    "moon_illumination","moonrise","moonset",
                    "sunrise","sunset","DewPointC","FeelsLikeC",
                    "HeatIndexC","WindChillC","visibility","location"]
for feature in eliminationfeature:
    df = df.drop(feature, axis = 1)

#%% Adding Feature 
df = df.astype(float)
df['Sin_windspeedKmph'] = np.sin(df['windspeedKmph'])
df['Cos_windspeedKmph'] = np.cos(df['windspeedKmph'])

df.index = df.index.rename('DateTime')
df = pd.merge(df, rng, on="DateTime", how="outer")

Target = pd.merge(Target, rng, on="DateTime", how="outer")

#%% CUT-OFF VE YANLIŞ VERİLER

df = df.loc[(df['DateTime'] <= "2021-07-27 00:00:00") | (df['DateTime'] >= "2021-07-29 23:00:00")]
df = df.loc[(df['DateTime'] <= "2021-07-30 09:00:00") | (df['DateTime'] >= "2021-08-06 10:00:00")]
df = df.loc[(df['DateTime'] <= "2021-08-07 14:00:00") | (df['DateTime'] >= "2021-08-07 22:00:00")]
df = df.loc[(df['DateTime'] <= "2021-08-09 12:00:00") | (df['DateTime'] >= "2021-08-09 23:00:00")]
df = df.loc[(df['DateTime'] <= "2021-08-10 12:00:00") | (df['DateTime'] >= "2021-08-19 20:00:00")]
df = df.loc[(df['DateTime'] <= "2021-08-28 16:00:00") | (df['DateTime'] >= "2021-09-01 17:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-07 09:00:00") | (df['DateTime'] >= "2021-09-07 19:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-08 15:00:00") | (df['DateTime'] >= "2021-09-08 15:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-13 00:00:00") | (df['DateTime'] >= "2021-09-14 09:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-15 05:00:00") | (df['DateTime'] >= "2021-09-15 09:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-21 11:00:00") | (df['DateTime'] >= "2021-09-21 23:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-30 08:00:00") | (df['DateTime'] >= "2021-09-30 17:00:00")]
df = df.loc[(df['DateTime'] <= "2021-10-22 10:00:00") | (df['DateTime'] >= "2021-10-23 04:00:00")]
df = df.loc[(df['DateTime'] <= "2021-10-26 06:00:00") | (df['DateTime'] >= "2021-10-26 14:00:00")]
df = df.loc[(df['DateTime'] <= "2021-10-27 03:00:00") | (df['DateTime'] >= "2021-10-27 16:00:00")]
df = df.loc[(df['DateTime'] <= "2021-10-29 10:00:00") | (df['DateTime'] >= "2021-10-29 16:00:00")]
df = df.loc[(df['DateTime'] <= "2021-10-30 07:00:00") | (df['DateTime'] >= "2021-10-30 11:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-06 05:00:00") | (df['DateTime'] >= "2021-11-06 14:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-08 03:00:00") | (df['DateTime'] >= "2021-11-08 08:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-08 23:00:00") | (df['DateTime'] >= "2021-11-09 18:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-10 13:00:00") | (df['DateTime'] >= "2021-11-10 17:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-11 06:00:00") | (df['DateTime'] >= "2021-11-11 15:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-12 00:00:00") | (df['DateTime'] >= "2021-11-12 03:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-15 18:00:00") | (df['DateTime'] >= "2021-11-16 01:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-23 03:00:00") | (df['DateTime'] >= "2021-11-23 17:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-25 05:00:00") | (df['DateTime'] >= "2021-11-25 17:00:00")]
df = df.loc[(df['DateTime'] <= "2021-11-27 06:00:00") | (df['DateTime'] >= "2021-11-30 22:00:00")]
df = df.loc[(df['DateTime'] <= "2021-12-10 14:00:00") | (df['DateTime'] >= "2021-12-10 15:00:00")]

df = df.loc[(df['DateTime'] <= "2021-09-03 20:00:00") | (df['DateTime'] >= "2021-09-05 20:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-06 11:00:00") | (df['DateTime'] >= "2021-09-06 15:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-09 04:00:00") | (df['DateTime'] >= "2021-09-11 00:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-13 00:00:00") | (df['DateTime'] >= "2021-09-16 13:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-17 00:00:00") | (df['DateTime'] >= "2021-09-17 23:00:00")]
df = df.loc[(df['DateTime'] <= "2021-09-19 20:00:00") | (df['DateTime'] >= "2021-09-23 23:00:00")]
df = df.loc[(df['DateTime'] <= "2021-12-28 00:00:00") | (df['DateTime'] >= "2021-12-30 23:00:00")]


Target = Target.loc[(Target['DateTime'] <= "2021-07-27 00:00:00") | (Target['DateTime'] >= "2021-07-29 23:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-07-30 09:00:00") | (Target['DateTime'] >= "2021-08-06 10:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-08-07 14:00:00") | (Target['DateTime'] >= "2021-08-07 22:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-08-09 12:00:00") | (Target['DateTime'] >= "2021-08-09 23:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-08-10 12:00:00") | (Target['DateTime'] >= "2021-08-19 20:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-08-28 16:00:00") | (Target['DateTime'] >= "2021-09-01 17:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-07 09:00:00") | (Target['DateTime'] >= "2021-09-07 19:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-08 15:00:00") | (Target['DateTime'] >= "2021-09-08 15:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-13 00:00:00") | (Target['DateTime'] >= "2021-09-14 09:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-15 05:00:00") | (Target['DateTime'] >= "2021-09-15 09:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-21 11:00:00") | (Target['DateTime'] >= "2021-09-21 23:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-30 08:00:00") | (Target['DateTime'] >= "2021-09-30 17:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-10-22 10:00:00") | (Target['DateTime'] >= "2021-10-23 04:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-10-26 06:00:00") | (Target['DateTime'] >= "2021-10-26 14:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-10-27 03:00:00") | (Target['DateTime'] >= "2021-10-27 16:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-10-29 10:00:00") | (Target['DateTime'] >= "2021-10-29 16:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-10-30 07:00:00") | (Target['DateTime'] >= "2021-10-30 11:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-06 05:00:00") | (Target['DateTime'] >= "2021-11-06 14:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-08 03:00:00") | (Target['DateTime'] >= "2021-11-08 08:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-08 23:00:00") | (Target['DateTime'] >= "2021-11-09 18:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-10 13:00:00") | (Target['DateTime'] >= "2021-11-10 17:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-11 06:00:00") | (Target['DateTime'] >= "2021-11-11 15:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-12 00:00:00") | (Target['DateTime'] >= "2021-11-12 03:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-15 18:00:00") | (Target['DateTime'] >= "2021-11-16 01:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-23 03:00:00") | (Target['DateTime'] >= "2021-11-23 17:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-25 05:00:00") | (Target['DateTime'] >= "2021-11-25 17:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-11-27 06:00:00") | (Target['DateTime'] >= "2021-11-30 22:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-12-10 14:00:00") | (Target['DateTime'] >= "2021-12-10 15:00:00")]

Target = Target.loc[(Target['DateTime'] <= "2021-09-03 20:00:00") | (Target['DateTime'] >= "2021-09-05 20:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-06 11:00:00") | (Target['DateTime'] >= "2021-09-06 15:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-09 04:00:00") | (Target['DateTime'] >= "2021-09-11 00:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-13 00:00:00") | (Target['DateTime'] >= "2021-09-16 13:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-17 00:00:00") | (Target['DateTime'] >= "2021-09-17 23:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-09-19 20:00:00") | (Target['DateTime'] >= "2021-09-23 23:00:00")]
Target = Target.loc[(Target['DateTime'] <= "2021-12-28 00:00:00") | (Target['DateTime'] >= "2021-12-30 23:00:00")]

df = df.set_index('DateTime')
Target = Target.set_index('DateTime')

Target = Target.dropna(axis='rows')

yeni = pd.DataFrame(df['windspeedKmph'])
yeni["WindGustKmph"] = df["WindGustKmph"]
yeni["Rüzgar"] = Target["Rüzgar"]
yeni["WinddirDegree"] = df["winddirDegree"]

#%% create time series plot
for column in df.columns:
    df[column].plot(figsize=(15, 6))
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.title('Time Series Plot of {}'.format(column.upper()))
    plt.show()

#%% create the Histogram plot
for column in df.columns:
    sns.displot(df[column],kde = True)
    plt.title(column, fontsize = 15)
    plt.show()

#%% create the boxplot
for feature in df.columns:
    ax = sns.boxplot(x = df[feature])
    ax.set_title("{} boxplot".format(feature), fontsize =20, pad = 20)
    plt.show()

#%% Handling outliers

# def plot_anomalies(df, feature, featurecolumn):
#     fig, ax = plt.subplots(figsize=(10,6))
#     a = df.loc[df[feature] == 0, [featurecolumn]] #anomaly
#     ax.plot(df.index, df[featurecolumn],label = 'Normal')
#     ax.scatter(a.index, a[featurecolumn], color='red', label = 'Anomaly')
#     plt.title(feature+" Outlier Plot")
#     plt.ylabel("Values")
#     plt.xlabel("Time")
#     plt.legend()
#     plt.show()
    
# def handing_outlier(df, feature):
#     EE_model = EllipticEnvelope(contamination = 0.02)
#     outliers = EE_model.fit_predict(df[[feature]])
#     df["EE_outliers"] = outliers
#     df["EE_outliers"] = df["EE_outliers"].apply(lambda x: str(0) if x == -1 else str(1))
#     df["EE_scores"] = EE_model.score_samples(df[[feature]])
#     df["EE_outliers"] = df["EE_outliers"].astype(float)
#     print(df["EE_outliers"].value_counts())
#     plot_anomalies(df,"EE_outliers", feature)
#     df = df.drop('EE_outliers', axis = 1)
#     df = df.drop('EE_scores', axis = 1)
#     return df

#%% Analyzing outlier and drop order quantity on the data
# print(df.shape)
# df = handing_outlier(df,'windspeedKmph')
# for i in range(len(df)):
#     if df["windspeedKmph"].iloc[i] > 13:
#         df["windspeedKmph"].iloc[i] = 13
        
# df = handing_outlier(df,'WindGustKmph')

# for i in range(len(df)):
#     if df["WindGustKmph"].iloc[i] > 35:
#         df["WindGustKmph"].iloc[i] = 35
        
# print(df.shape)

#%% TRAINING AND VALIDATION SET
df = df.values
Target = Target.values

k = 0
n_splits=5
    
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=n_splits, test_size=None)
print(tscv)
RMSE = np.zeros(n_splits)
MAE = np.zeros(n_splits)
for train_index, test_index in tscv.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)    
    X_train, X_test = df[train_index], df[test_index]
    y_train, y_test = Target[train_index], Target[test_index]    
    Xgbpred = XGBoostRegression(X_train,y_train,X_test)
    RMSE[k] = rmse(Xgbpred, y_test)
    MAE[k] = mean_absolute_error(y_test, Xgbpred)
    train_test_plot(y_test,Xgbpred)
    k = k + 1
    if k == 4:
        break
    
#%% TEST SET
end_date=today
hist_weather_data2 = retrieve_hist_data('beb3f237e0274253b3764922210712', ['41.2627,28.1413'],
                                        yesterday, end_date, 
                                        1, location_label = False, 
                                        export_csv = False, store_df = True)
X_test2 = hist_weather_data2[0]
X_test2 = X_test2[24:]
X_test2['hour'] = pd.DatetimeIndex(X_test2['date_time']).hour
X_test2['month'] = pd.DatetimeIndex(X_test2['date_time']).month
X_test2['dayofweek'] = pd.DatetimeIndex(X_test2['date_time']).dayofweek

X_test2 = X_test2.set_index('date_time')

#%% Feature Elimination
eliminationfeature=["maxtempC","mintempC","sunHour","uvIndex",
                    "moon_illumination","moonrise","moonset",
                    "sunrise","sunset","DewPointC","FeelsLikeC",
                    "HeatIndexC","WindChillC","visibility","location"]
for feature in eliminationfeature:
    X_test2 = X_test2.drop(feature, axis = 1)
#%% Adding Feature 
X_test2 = X_test2.astype(float)
X_test2['Sin_windspeedKmph'] = np.sin(X_test2['windspeedKmph'])
X_test2['Cos_windspeedKmph'] = np.cos(X_test2['windspeedKmph'])
X_test2 = X_test2.values

Xgbpred = XGBoostRegression(X_train,y_train,X_test2)

for ij in range(len(Xgbpred)):
    if Xgbpred[ij] > installedcapacity:
        Xgbpred[ij] = installedcapacity
    elif Xgbpred[ij] < 0:
        Xgbpred[ij]=0
        
x = range(len(Xgbpred))
ax = plt.figure().add_subplot(111)
ax.plot(x, list(Xgbpred), linewidth=1.5)
ax.set_ylim(0, installedcapacity)
ax.set_title("{} ISTANBUL RES".format(end_date), fontsize=10)
ax.set_xlabel('Hours', fontsize=12)
ax.set_ylabel('MWh', fontsize=12)
ax.grid(True)

Xgbpred2 = Xgbpred.copy()
Xgbpred2 = pd.DataFrame(Xgbpred2)
Xgbpred2.to_excel('ISTANBULRES.xlsx')
