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


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
presentday = datetime.now()
tomorrow = (presentday + timedelta(1)).strftime('%Y-%m-%d')


start = pd.to_datetime('2021-01-01')
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
    gsearch1.fit(X_train,y_train)
    print (gsearch1.best_params_)
    print('best score')
    print (gsearch1.best_score_)
    Xgbpred = gsearch1.predict(X_test)

    return Xgbpred

#%% DATA PROCESSING

#URETIM
Target = uretim.gerceklesen(baslangic_tarihi = '2021-01-01', bitis_tarihi = '2021-09-06', santral_id = "2382")
Target['DateTime'] = pd.to_datetime(Target.Tarih) + Target.Saat.astype('timedelta64[h]')
Target = Target.set_index('DateTime')
Target = pd.DataFrame(Target['Rüzgar'])


frequency=1
start_date = '01-JAN-2021'
end_date = '06-SEP-2021'
api_key = 'beb3f237e0274253b3764922210712'
location_list = ['41.173274,28.303366']
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

df = df.astype(float)

df['Sin_windspeedKmph'] = np.sin(df['windspeedKmph'])
df['Cos_windspeedKmph'] = np.cos(df['windspeedKmph'])

# df['wind_scalar^2'] = df['windspeedKmph']**2
# df['wind_scalar^3'] = df['windspeedKmph']**3

df.index = df.index.rename('DateTime')
df = pd.merge(df, rng, on="DateTime", how="outer")

"Correlation matrix"
import seaborn as sns
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, vmax=.8, square=True, cmap='coolwarm', fmt='.2f')
plt.xticks(rotation=90)
plt.show()

# "Creating correlation matrix features without target value"
# olddf = df.columns
# print("Before eliminating shape: ",df.shape)
# corr_matrix = df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# #Keeping at least any feature in data.
# to_drop = [column for column in upper.columns if any(upper[column] >= 0.92)]
# # Dropping index of feature columns with correlation greater than 0.95
# df = df.drop(df[to_drop], axis=1)
# newdf = df.columns
# print("After eliminating shape: ",df.shape)
# print("Eliminating features:\n",list(set(olddf) - set(newdf)))

Target = pd.merge(Target, rng, on="DateTime", how="outer")

Target = Target.sort_values(by='DateTime', ascending=True)

df = df.loc[(df['DateTime'] < "2021-01-01 08:00:00") | (df['DateTime'] > "2021-01-01 15:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-05 00:00:00") | (df['DateTime'] > "2021-01-05 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-06 18:00:00") | (df['DateTime'] > "2021-01-06 22:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-08 14:00:00") | (df['DateTime'] > "2021-01-09 09:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-10 03:00:00") | (df['DateTime'] > "2021-01-10 14:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-12 03:00:00") | (df['DateTime'] > "2021-01-13 14:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-15 00:00:00") | (df['DateTime'] > "2021-01-15 13:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-16 00:00:00") | (df['DateTime'] > "2021-01-16 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-24 18:00:00") | (df['DateTime'] > "2021-01-24 20:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-25 17:00:00") | (df['DateTime'] > "2021-01-26 02:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-26 09:00:00") | (df['DateTime'] > "2021-01-26 18:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-27 01:00:00") | (df['DateTime'] > "2021-01-27 03:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-30 11:00:00") | (df['DateTime'] > "2021-01-30 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-01-31 11:00:00") | (df['DateTime'] > "2021-01-31 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-02 03:00:00") | (df['DateTime'] > "2021-02-02 07:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-06 10:00:00") | (df['DateTime'] > "2021-02-06 13:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-07 19:00:00") | (df['DateTime'] > "2021-02-08 06:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-08 17:00:00") | (df['DateTime'] > "2021-02-08 19:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-09 08:00:00") | (df['DateTime'] > "2021-02-09 20:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-10 08:00:00") | (df['DateTime'] > "2021-02-10 16:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-11 17:00:00") | (df['DateTime'] > "2021-02-13 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-16 22:00:00") | (df['DateTime'] > "2021-02-17 01:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-19 07:00:00") | (df['DateTime'] > "2021-02-19 15:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-21 00:00:00") | (df['DateTime'] > "2021-02-21 16:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-24 05:00:00") | (df['DateTime'] > "2021-02-24 07:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-25 00:00:00") | (df['DateTime'] > "2021-02-25 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-27 19:00:00") | (df['DateTime'] > "2021-02-27 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-02-28 14:00:00") | (df['DateTime'] > "2021-02-28 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-03 12:00:00") | (df['DateTime'] > "2021-03-03 20:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-06 00:00:00") | (df['DateTime'] > "2021-03-06 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-07 17:00:00") | (df['DateTime'] > "2021-03-07 21:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-15 17:00:00") | (df['DateTime'] > "2021-03-16 20:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-17 10:00:00") | (df['DateTime'] > "2021-03-17 20:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-20 19:00:00") | (df['DateTime'] > "2021-03-21 01:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-22 07:00:00") | (df['DateTime'] > "2021-03-22 14:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-24 02:00:00") | (df['DateTime'] > "2021-03-24 10:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-29 04:00:00") | (df['DateTime'] > "2021-03-29 12:00:00")]
df = df.loc[(df['DateTime'] < "2021-03-30 05:00:00") | (df['DateTime'] > "2021-03-30 11:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-01 00:00:00") | (df['DateTime'] > "2021-04-01 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-04 00:00:00") | (df['DateTime'] > "2021-04-05 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-07 06:00:00") | (df['DateTime'] > "2021-04-07 08:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-07 17:00:00") | (df['DateTime'] > "2021-04-07 19:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-08 13:00:00") | (df['DateTime'] > "2021-04-08 15:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-14 00:00:00") | (df['DateTime'] > "2021-04-14 23:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-17 23:00:00") | (df['DateTime'] > "2021-04-20 19:00:00")]
df = df.loc[(df['DateTime'] < "2021-04-21 16:00:00") | (df['DateTime'] > "2021-04-21 17:00:00")]
df = df.loc[(df['DateTime'] < "2021-05-09 16:00:00") | (df['DateTime'] > "2021-05-10 08:00:00")]

Target = Target.loc[(Target['DateTime'] < "2021-01-01 08:00:00") | (Target['DateTime'] > "2021-01-01 15:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-05 00:00:00") | (Target['DateTime'] > "2021-01-05 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-06 18:00:00") | (Target['DateTime'] > "2021-01-06 22:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-08 14:00:00") | (Target['DateTime'] > "2021-01-09 09:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-10 03:00:00") | (Target['DateTime'] > "2021-01-10 14:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-12 03:00:00") | (Target['DateTime'] > "2021-01-13 14:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-15 00:00:00") | (Target['DateTime'] > "2021-01-15 13:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-16 00:00:00") | (Target['DateTime'] > "2021-01-16 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-24 18:00:00") | (Target['DateTime'] > "2021-01-24 20:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-25 17:00:00") | (Target['DateTime'] > "2021-01-26 02:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-26 09:00:00") | (Target['DateTime'] > "2021-01-26 18:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-27 01:00:00") | (Target['DateTime'] > "2021-01-27 03:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-30 11:00:00") | (Target['DateTime'] > "2021-01-30 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-01-31 11:00:00") | (Target['DateTime'] > "2021-01-31 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-02 03:00:00") | (Target['DateTime'] > "2021-02-02 07:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-06 10:00:00") | (Target['DateTime'] > "2021-02-06 13:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-07 19:00:00") | (Target['DateTime'] > "2021-02-08 06:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-08 17:00:00") | (Target['DateTime'] > "2021-02-08 19:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-09 08:00:00") | (Target['DateTime'] > "2021-02-09 20:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-10 08:00:00") | (Target['DateTime'] > "2021-02-10 16:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-11 17:00:00") | (Target['DateTime'] > "2021-02-13 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-16 22:00:00") | (Target['DateTime'] > "2021-02-17 01:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-19 07:00:00") | (Target['DateTime'] > "2021-02-19 15:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-21 00:00:00") | (Target['DateTime'] > "2021-02-21 16:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-24 05:00:00") | (Target['DateTime'] > "2021-02-24 07:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-25 00:00:00") | (Target['DateTime'] > "2021-02-25 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-27 19:00:00") | (Target['DateTime'] > "2021-02-27 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-02-28 14:00:00") | (Target['DateTime'] > "2021-02-28 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-03 12:00:00") | (Target['DateTime'] > "2021-03-03 20:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-06 00:00:00") | (Target['DateTime'] > "2021-03-06 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-07 17:00:00") | (Target['DateTime'] > "2021-03-07 21:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-15 17:00:00") | (Target['DateTime'] > "2021-03-16 20:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-17 10:00:00") | (Target['DateTime'] > "2021-03-17 20:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-20 19:00:00") | (Target['DateTime'] > "2021-03-21 01:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-22 07:00:00") | (Target['DateTime'] > "2021-03-22 14:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-24 02:00:00") | (Target['DateTime'] > "2021-03-24 10:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-29 04:00:00") | (Target['DateTime'] > "2021-03-29 12:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-03-30 05:00:00") | (Target['DateTime'] > "2021-03-30 11:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-01 00:00:00") | (Target['DateTime'] > "2021-04-01 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-04 00:00:00") | (Target['DateTime'] > "2021-04-05 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-07 06:00:00") | (Target['DateTime'] > "2021-04-07 08:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-07 17:00:00") | (Target['DateTime'] > "2021-04-07 19:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-08 13:00:00") | (Target['DateTime'] > "2021-04-08 15:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-14 00:00:00") | (Target['DateTime'] > "2021-04-14 23:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-17 23:00:00") | (Target['DateTime'] > "2021-04-20 19:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-04-21 16:00:00") | (Target['DateTime'] > "2021-04-21 17:00:00")]
Target = Target.loc[(Target['DateTime'] < "2021-05-09 16:00:00") | (Target['DateTime'] > "2021-05-10 08:00:00")]


df = df.set_index('DateTime')
Target = Target.set_index('DateTime')


# yeni = pd.DataFrame(df['windspeedKmph'])
# yeni["WindGustKmph"] = df["WindGustKmph"]
# yeni["winddirDegree"] = df["winddirDegree"]
# yeni["Rüzgar"] = Target["Rüzgar"]

# yeni['DateTime'] = rng
# yeni = yeni.set_index('DateTime')

# plt.plot(yeni['Rüzgar'].values, linewidth=1.5, label="Production")
# plt.plot(yeni['windspeedKmph'].values, linewidth=1.5, label="windspeedKmph")
# plt.plot(yeni['WindGustKmph'].values, linewidth=1.5, label="WindGustKmph")
# plt.legend(loc='best',fancybox=True, shadow=True)



#%% TRAINING AND VALIDATION SET
df = df.values
Target = Target.values

k = 0
n_splits=5

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
start_date = yesterday
end_date = today
api_key = 'beb3f237e0274253b3764922210712'
location_list = ['41.1732740,28.3033660']

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


X_test2 = X_test2.astype(float)

X_test2['Sin_windspeedKmph'] = np.sin(X_test2['windspeedKmph'])
X_test2['Cos_windspeedKmph'] = np.cos(X_test2['windspeedKmph'])

# X_test2['wind_scalar^2'] = X_test2['windspeedKmph']**2
# X_test2['wind_scalar^3'] = X_test2['windspeedKmph']**3
X_test2 = X_test2.values

Xgbpred = XGBoostRegression(X_train,y_train,X_test2)

for ij in range(len(Xgbpred)):
    if Xgbpred[ij] > 3.00:
        Xgbpred[ij]=3
    elif Xgbpred[ij] < 0:
        Xgbpred[ij]=0

x = range(len(Xgbpred))
ax = plt.figure().add_subplot(111)
ax.plot(x, list(Xgbpred), linewidth=1.5)
ax.set_ylim(0, 3)
ax.set_title("{} SAKARBAYIR".format(end_date), fontsize=10)
ax.set_xlabel('Hours', fontsize=12)
ax.set_ylabel('MWh', fontsize=12)
ax.grid(True)

Xgbpred2 = Xgbpred.copy()
Xgbpred2 = pd.DataFrame(Xgbpred2)
Xgbpred2.to_excel('SAKARBAYIR.xlsx')




