from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from wwo_hist import retrieve_hist_data
import pandas as pd
import xgboost
import numpy as np
import lightgbm as lgb
from seffaflik.elektrik import uretim
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#%% IMPORT EPIAS AND WORLDWEATHER
today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
presentday = datetime.now()
tomorrow = (presentday + timedelta(1)).strftime('%Y-%m-%d')

tomorrow2 = (presentday + timedelta(2)).strftime('%Y-%m-%d')

#URETIM
Target = uretim.gerceklesen(baslangic_tarihi = '2019-10-02', bitis_tarihi = today, santral_id = "1833")
Target['DateTime'] = pd.to_datetime(Target.Tarih) + Target.Saat.astype('timedelta64[h]')
Target = Target.set_index('DateTime')
Target = pd.DataFrame(Target['Rüzgar'])

#EAK
# eak2 = uretim.eak(baslangic_tarihi='2019-10-02', bitis_tarihi=tomorrow, organizasyon_eic = '40X0000000081191')
# eak2 = pd.DataFrame(eak2['Rüzgar'])

# frequency=1
# start_date = '02-OCT-2019'
# end_date = tomorrow
# api_key = 'd4a0a860cee843f6b19124114211606'
# location_list = ['36.42,36.11']
# hist_weather_data = retrieve_hist_data(api_key, location_list,
#                                         start_date, end_date, 
#                                         frequency, location_label = False, 
#                                         export_csv = True, store_df = True)

df = pd.read_csv('36.42,36.11.csv')

df = df.drop('date_time', axis = 1)
df = df.drop('maxtempC', axis = 1)
df = df.drop('mintempC', axis = 1)
df = df.drop('totalSnow_cm', axis = 1)
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
df = df.drop('cloudcover', axis = 1)
df = df.drop('visibility', axis = 1)
df = df.drop('location', axis = 1)
df = df.drop('precipMM', axis = 1)
df = df.drop('humidity', axis = 1)
df = df.drop('tempC', axis = 1)
df = df.drop('pressure', axis = 1)

start = pd.to_datetime('2019-10-02')
end = pd.to_datetime(tomorrow2)
rng = pd.date_range(start, end, freq='H')
len(rng)
rng = rng[0:-1]
len(rng)

df['DateTime'] = rng
df = df.set_index('DateTime')

df = pd.merge(df, Target, left_index=True, right_index=True, how="outer")

df= df[df['Rüzgar'] >= 1]

Target = uretim.gerceklesen(baslangic_tarihi = '2019-10-02', bitis_tarihi = today, santral_id = "1833")

# df = pd.merge(df, Target, on="DateTime", how="outer")

# eak2['DateTime'] = rng
# df = pd.merge(df, eak2, on="DateTime", how="left")


df = df.drop(df.index[-48:-24])

#%% SPLITTING TRAIN AND TEST

splitting = -24
X_train = df[0:splitting]
y_train = Target

X_test = df[splitting:]

#%% STANDARD SCALING
# scale input data
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

# scale input data
# from sklearn import preprocessing

# col_names = df.columns
# features = df[col_names]

# scaler = MinMaxScaler().fit(features.values)
# features = scaler.transform(X_train.values)
# scaled_features = pd.DataFrame(features, columns = col_names)
# scaled_features.head()

# sc_X = MinMaxScaler()
# sc_y = MinMaxScaler().fit(df.values)

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)
# y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

#%% Initialize CatBoostRegressor

def CatBoost(X_train,y_train,X_test):
    grid = {'iterations': [1000],
        'learning_rate': [0.1],
        'depth': [4, 8],
        'l2_leaf_reg': [0.5, 1]}
    model1 = GridSearchCV(estimator = CatBoostRegressor(), param_grid = grid, n_jobs = -1, verbose = 2)
    model1.fit(X_train, y_train)
    Catpreds = model1.predict(X_test)
    return Catpreds

#%% Initialize MultilayerPerceptronRegressor

def MultiLayerPerceptron(X_train,y_train,X_test):
    param_list = {'solver': ['lbfgs', 'adam'], 'max_iter': np.arange(100, 200, 50), 'alpha': [0.0001] , 'hidden_layer_sizes': np.arange(10, 15)}
    # param_list = {"hidden_layer_sizes": [1,50], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}
    mlp = GridSearchCV(estimator = MLPRegressor(), param_grid = param_list, n_jobs = -1, verbose = 2)
    mlp.fit(X_train, y_train)
    Mlppreds = mlp.predict(X_test)
    return Mlppreds

#%% Initialize LightGBMRegressor
def LightGBMRegression(X_train,y_train,X_test):
    hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 1000,
    "n_estimators": 1000
    }
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train, y_train)
    Lightpred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    return Lightpred

#%% Initialize XGBoostRegression
def XGBoostRegression(X_train,y_train,X_test):

    # for tuning parameters
    parameters_for_testing = {
    'colsample_bytree':[0.8],
    'min_child_weight':[1.5, 6, 10],
    'learning_rate':[0.1],
    'max_depth':[3,5],
    'n_estimators':[1000],
    'subsample':[0.8]  
    }

                    
    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
                                     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs = -1, verbose = 2, scoring='neg_mean_squared_error')
    gsearch1.fit(X_train,y_train)
    print (gsearch1.best_params_)
    print('best score')
    print (gsearch1.best_score_)
    Xgbpred = gsearch1.predict(X_test)
    
    return Xgbpred

#%% PREDICTION

# for inverse transformation
Catpreds = CatBoost(X_train,y_train,X_test)
Catpreds = sc_y.inverse_transform(Catpreds)
for i in range(len(Catpreds)):
    if Catpreds[i] < 0:
        Catpreds[i] = 0
    else:
        continue

# for inverse transformation
Mlppreds = MultiLayerPerceptron(X_train,y_train,X_test)
Mlppreds = sc_y.inverse_transform(Mlppreds)
for i in range(len(Mlppreds)):
    if Mlppreds[i] < 0:
        Mlppreds[i] = 0
    else:
        continue

# for inverse transformation
Lightpred = LightGBMRegression(X_train,y_train,X_test)
Lightpred = sc_y.inverse_transform(Lightpred)
for i in range(len(Lightpred)):
    if Lightpred[i] < 0:
        Lightpred[i] = 0
    else:
        continue

# for inverse transformation
Xgbpred = XGBoostRegression(X_train,y_train,X_test)
Xgbpred = sc_y.inverse_transform(Xgbpred)
for i in range(len(Xgbpred)):
    if Xgbpred[i] < 0:
        Xgbpred[i] = 0
    else:
        continue

#%% PLOT

fig, ax = plt.subplots(figsize=(10,8))

idx = pd.date_range(today, tomorrow, freq = 'H')
idx = idx[0:-1]
idx2 = pd.Series(Catpreds,  index = idx)
idx3 = pd.Series(Mlppreds,  index = idx)
idx4 = pd.Series(Lightpred,  index = idx)
idx5 = pd.Series(Xgbpred,  index = idx)


hours = mdates.HourLocator(interval = 1)
h_fmt = mdates.DateFormatter('%H:%M:%S')

ax.plot(idx2.index, idx2.values, linewidth = 1.5, label="CatBoost")
ax.plot(idx3.index, idx3.values, linewidth = 1.5, label="MLP")
ax.plot(idx4.index, idx4.values, linewidth = 1.5, label="LightGBM")
ax.plot(idx5.index, idx5.values, linewidth = 1.5, label="XGBoost")

plt.title("{} Çerçikaya".format(tomorrow))
plt.ylabel('MW/h')
plt.legend(loc='best',fancybox=True, shadow=True)
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(h_fmt)
plt.grid(True)
fig.autofmt_xdate()
plt.show()

