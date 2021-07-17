from sklearn.linear_model import Lasso, Ridge
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from WorldWeatherPy import DetermineListOfAttributes
from WorldWeatherPy import HistoricalLocationWeather
from wwo_hist import retrieve_hist_data
from sklearn import ensemble
import pandas as pd
import xgboost
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE 
from seffaflik.elektrik import santraller, tuketim, uretim, yekdem
from seffaflik.elektrik.piyasalar import dengesizlik, dgp, genel, gip, gop, ia, yanhizmetler

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#%% IMPORT EPIAS AND WORLDWEATHER

meteo = pd.read_excel('MutluMeteo.xls')
meteo = meteo.drop('DateTime', axis = 1)


Target = uretim.gerceklesen(baslangic_tarihi='2020-09-01', bitis_tarihi='2021-04-29', santral_id="2570")
Target['DateTime'] = pd.to_datetime(Target.Tarih) + Target.Saat.astype('timedelta64[h]')
Target = Target.drop('Tarih', axis = 1)
Target = Target.drop('Saat', axis = 1)
Target = Target.set_index('DateTime')
Target = pd.DataFrame(Target['Rüzgar'])

frequency=1
start_date = '01-SEP-2020'
end_date = '30-APR-2021'
api_key = '6e3be25b74324173a8172756210704'
location_list = ['38.15,32.54']
hist_weather_data = retrieve_hist_data(api_key,location_list,start_date,end_date,frequency,location_label = False,export_csv = True,store_df = True)
df = pd.read_csv('38.15,32.54.csv')

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

start = pd.to_datetime('2020-09-01')
end = pd.to_datetime('2021-05-01')
rng = pd.date_range(start, end, freq='H')
len(rng)
rng = rng[0:-1]
len(rng)

# Target['DateTime'] = rng
df['DateTime'] = rng
meteo['DateTime'] = rng

# Target = Target.set_index('DateTime')
df = df.set_index('DateTime')
meteo = meteo.set_index('DateTime')

df = pd.merge(df, meteo, on="DateTime", how="left")
df = df.drop('p50', axis = 1)


# df = pd.merge(df, Target, on="DateTime", how="left")



# df = df.drop(df.index[-48:-24])


# df = pd.merge(df, Target, on="DateTime", how="left")

# mask = Target['Rüzgar'] <= 1
# df['mask'] = list(map(int, mask))
# df = df.drop('Rüzgar', axis = 1)

# df['Sin'] = np.sin(pd.DatetimeIndex(df.index).hour)
# df['Cos'] = np.cos(pd.DatetimeIndex(df.index).hour)

df['Sin_windspeedKmph'] = np.sin(df['windspeedKmph'])
df['Cos_windspeedKmph'] = np.cos(df['windspeedKmph'])

df['Sin_winddirDegree'] = np.sin(df['winddirDegree'])
df['Cos_winddirDegree'] = np.cos(df['winddirDegree'])

# df['Week'] = pd.DatetimeIndex(df.index).dayofweek
# df['Hour'] = pd.DatetimeIndex(df.index).hour
# df['Month'] = pd.DatetimeIndex(df.index).month
# df['WeekXMonth'] = df['Week'] * df['Month']
# df['Sin'] = np.sin(pd.DatetimeIndex(df.index).hour)
# df['Cos'] = np.cos(pd.DatetimeIndex(df.index).hour)

df['wind_scalar^2'] = df['windspeedKmph']**2
df['wind_scalar^3'] = df['windspeedKmph']**3

df = df.drop(df.index[-48:-24])

# df['wind_scalarXMonth'] = df['windspeedKmph'] * df['Month']
# df['wind_scalar^2XMonth'] = df['windspeedKmph']**2 * df['Month']
# df['wind_scalar^3XMonth'] = df['windspeedKmph']**3 * df['Month']

# df['wind_scalarXHour'] = df['windspeedKmph'] * df['Hour']
# df['wind_scalar^2XHour'] = df['windspeedKmph']**2 * df['Hour']
# df['wind_scalar^3XHour'] = df['windspeedKmph']**3 * df['Hour']
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

# sc_X = MinMaxScaler()
# sc_y = MinMaxScaler()

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_y.fit_transform(X_test)
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

#%% Initialize AdaBoostRegressor

def AdaBoost(X_train,y_train,X_test):
    search_grid={'n_estimators': [600,800], 'learning_rate':[0.1]}
    ada = GridSearchCV(estimator = AdaBoostRegressor(), param_grid = search_grid, n_jobs = -1, verbose = 2)
    ada.fit(X_train, y_train)
    Adapreds = ada.predict(X_test)
    return Adapreds

#%% Initialize MultilayerPerceptronRegressor

def MultiLayerPerceptron(X_train,y_train,X_test):
    param_list = {'solver': ['lbfgs', 'adam'], 'max_iter': np.arange(100, 200, 50), 'alpha': [0.0001] , 'hidden_layer_sizes': np.arange(10, 15)}
    # param_list = {"hidden_layer_sizes": [1,50], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}
    mlp = GridSearchCV(estimator = MLPRegressor(), param_grid = param_list, n_jobs = -1, verbose = 2)
    mlp.fit(X_train, y_train)
    Mlppreds = mlp.predict(X_test)
    return Mlppreds

#%% Initialize RandomForestRegressor

def RandomForest(X_train,y_train,X_test):
    param_grid = {'max_depth': [80, 100], 'min_samples_leaf': [3, 4], 'min_samples_split': [8, 10], 'n_estimators': [100]}
    rf = GridSearchCV(estimator = RandomForestRegressor(), param_grid = param_grid, n_jobs = -1, verbose = 2)
    rf.fit(X_train, y_train)
    # rf = RandomForestRegressor(n_estimators=100)
    # rf = RandomForestRegressor(rf.best_params_['n_estimators'])
    # rf.fit(X_train, y_train)
    # sorted_idx = rf.feature_importances_.argsort()
    # plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    # Get predictions
    RFpreds = rf.predict(X_test)
    return RFpreds

#%% Initialize RidgeRegressor

def RidgeRegressor(X_train,y_train,X_test):
    params_Ridge = {'alpha': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100, 1000], "fit_intercept": [True, False], "solver": ['lsqr']}
    Ridge_GS = GridSearchCV(estimator = Ridge(), param_grid = params_Ridge, n_jobs=-1, verbose = 2)
    Ridge_GS.fit(X_train, y_train)
    # Ridge_GS.best_params_
    # Get predictions
    Ridgepreds = Ridge_GS.predict(X_test)
    return Ridgepreds

#%% Initialize LassoRegressor
def LassoRegressor(X_train,y_train,X_test):
    params_Lasso = {'alpha': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}
    Lasso1 = GridSearchCV(estimator = Lasso(), param_grid = params_Lasso, cv=5, n_jobs=-1, verbose = 2)
    Lasso1.fit(X_train, y_train)
    # Get predictions
    Lassopreds = Lasso1.predict(X_test)
    return Lassopreds

#%% Initialize GradientBoostingRegressor
def GradientBoosting(X_train,y_train,X_test):
    params = {'n_estimators': 100,'max_depth': 4,'learning_rate': 0.1,'subsample': 0.5,'max_depth': 4}
    grid_GBR = ensemble.GradientBoostingRegressor(**params)
    grid_GBR.fit(X_train, y_train)
    
    feature_importance = grid_GBR.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(df.columns)[sorted_idx])
    plt.title('Feature Importance Gradient Boosting')

    GBpreds = grid_GBR.predict(X_test)
    return GBpreds

#%% Initialize KNeighborsRegressor

def Knn(X_train,y_train,X_test):
    param = {'n_neighbors': np.arange(1, 101, 10), 'weights': ['uniform', 'distance']}
    knn = KNeighborsRegressor(metric='euclidean')
    model = GridSearchCV(estimator = knn, param_grid = param, n_jobs=-1, verbose=2)
    model.fit(X_train, y_train)
    # Get predictions
    knnpreds = model.predict(X_test)
    return knnpreds

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

#for tuning parameters
    parameters_for_testing = {
    'colsample_bytree':[0.4,0.8],
    'gamma':[0,0.03,0.1],
    'min_child_weight':[1.5,10],
    'learning_rate':[0.1],
    'max_depth':[3,5],
    'n_estimators':[1000],
    'reg_alpha':[1e-2],
    'reg_lambda':[1e-2],
    'subsample':[0.6]  
    }

                    
    xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
                                     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

    gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6, verbose=10,scoring='neg_mean_squared_error')
    gsearch1.fit(X_train,y_train)
    # print (gsearch1.best_params_)
    Xgbpred = gsearch1.predict(X_test)
    
    # best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
    #              gamma=0,                 
    #              learning_rate=0.07,
    #              max_depth=3,
    #              min_child_weight=1.5,
    #              n_estimators=10000,                                                                    
    #              reg_alpha=0.75,
    #              reg_lambda=0.45,
    #              subsample=0.6,
    #              seed=42)
    # best_xgb_model.fit(X_train,y_train)
    # Xgbpred = best_xgb_model.predict(X_test)
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
# RMSECatpreds = rmse(Catpreds, y_test.values)

# for inverse transformation
Adapreds = AdaBoost(X_train,y_train,X_test)
Adapreds = sc_y.inverse_transform(Adapreds)
for i in range(len(Adapreds)):
    if Adapreds[i] < 0:
        Adapreds[i] = 0
    else:
        continue
# RMSEAdapreds = rmse(Adapreds, y_test.values)

# for inverse transformation
RFpreds = RandomForest(X_train,y_train,X_test)
RFpreds = sc_y.inverse_transform(RFpreds)
for i in range(len(RFpreds)):
    if RFpreds[i] < 0:
        RFpreds[i] = 0
    else:
        continue
# RMSERFpreds = rmse(RFpreds, y_test.values)

# for inverse transformation
Ridgepreds = RidgeRegressor(X_train,y_train,X_test)
Ridgepreds = sc_y.inverse_transform(Ridgepreds)
for i in range(len(Ridgepreds)):
    if Ridgepreds[i] < 0:
        Ridgepreds[i] = 0
    else:
        continue
# RMSERidgepreds = rmse(Ridgepreds, y_test.values)

# for inverse transformation
Lassopreds = LassoRegressor(X_train,y_train,X_test)
Lassopreds = sc_y.inverse_transform(Lassopreds)
for i in range(len(Lassopreds)):
    if Lassopreds[i] < 0:
        Lassopreds[i] = 0
    else:
        continue
# RMSELassopreds = rmse(Lassopreds, y_test.values)

# for inverse transformation
GBpreds = GradientBoosting(X_train,y_train,X_test)
GBpreds = sc_y.inverse_transform(GBpreds)
for i in range(len(GBpreds)):
    if GBpreds[i] < 0:
        GBpreds[i] = 0
    else:
        continue
# RMSEGBpreds = rmse(GBpreds, y_test.values)

# for inverse transformation
knnpreds = Knn(X_train,y_train,X_test)
knnpreds = sc_y.inverse_transform(knnpreds)
for i in range(len(knnpreds)):
    if knnpreds[i] < 0:
        knnpreds[i] = 0
    else:
        continue
# RMSEknnpreds = rmse(knnpreds, y_test.values)

# for inverse transformation
Mlppreds = MultiLayerPerceptron(X_train,y_train,X_test)
Mlppreds = sc_y.inverse_transform(Mlppreds)
for i in range(len(Mlppreds)):
    if Mlppreds[i] < 0:
        Mlppreds[i] = 0
    else:
        continue
# RMSEMlppreds = rmse(Mlppreds, y_test.values)

# for inverse transformation
Lightpred = LightGBMRegression(X_train,y_train,X_test)
Lightpred = sc_y.inverse_transform(Lightpred)
for i in range(len(Lightpred)):
    if Lightpred[i] < 0:
        Lightpred[i] = 0
    else:
        continue
# RMSELightpreds = rmse(Lightpred, y_test.values)

# for inverse transformation
Xgbpred = XGBoostRegression(X_train,y_train,X_test)
Xgbpred = sc_y.inverse_transform(Xgbpred)
for i in range(len(Xgbpred)):
    if Xgbpred[i] < 0:
        Xgbpred[i] = 0
    else:
        continue
# RMSEXgbpreds = rmse(Xgbpred, y_test.values)

#%% PLOT

# x_ax = range(len(y_test))
# plt.plot(x_ax, list(y_test.values), linewidth=1.5, label="original")
# plt.plot(x_ax, list(Xgbpred), linewidth=1.5, label="predicted")
# # plt.plot(x_ax, list(meteo['p50'].tail(168).values), linewidth=1.5, label="Meteo")
# plt.title("Mutlu RES y-test and y-predicted data")
# plt.xlabel('Days')
# plt.ylabel('MW/h')
# plt.legend(loc='best',fancybox=True, shadow=True)
# plt.grid(True)
# plt.show()

# #%% RMSE Plotting

# x = ['CatBoost','Adaboost','RandomForest','Ridge','Lasso','GradientBoosting','kNN', 'MultiLayerPerceptron', 'LightGBM', 'XGBoost']
# y = [RMSECatpreds, RMSEAdapreds, RMSERFpreds, RMSERidgepreds, RMSELassopreds, RMSEGBpreds, RMSEknnpreds, RMSEMlppreds, RMSELightpreds, RMSEXgbpreds]
# newdf = pd.DataFrame({"Methods":x, "Results":y})
# newdf_sorted = newdf.sort_values('Results',ascending=False)
# plt.barh('Methods', 'Results',data=newdf_sorted)
# plt.title('Error Prediction')
# plt.show()

# Ridgepreds = Ridgepreds.drop()

# Catpreds = pd.DataFrame(Catpreds)
# Adapreds = pd.DataFrame(Adapreds)
# RFpreds = pd.DataFrame(RFpreds)
# Ridgepreds = pd.DataFrame(Ridgepreds)
# Lassopreds = pd.DataFrame(Lassopreds)
# GBpreds = pd.DataFrame(GBpreds)
# knnpreds = pd.DataFrame(knnpreds)
# Mlppreds = pd.DataFrame(Mlppreds)
# Lightpred = pd.DataFrame(Lightpred)
# Xgbpred = pd.DataFrame(Xgbpred)


# y = [Catpreds, Adapreds, RFpreds, Ridgepreds, Lassopreds, GBpreds, knnpreds, Mlppreds, Lightpred, Xgbpred]
# i=0
# while True:
#     if i == 10:
#         break
#     else:
#     pd.concat([y[i], y[i+1]], axis=1, join="inner")
#     i=i+2

    

# MutluRES = MutluRES.set_axis(['CatBoost','Adaboost','RandomForest','Ridge','Lasso','GradientBoosting','kNN', 'MultiLayerPerceptron', 'LightGBM', 'XGBoost'], axis=1, inplace=False)
# y = {'CatBoost': [Catpreds], 'Adaboost': Adapreds, 'RandomForest': RFpreds[0], 'Ridge': Ridgepreds, 'Lasso': Lassopreds, 'GradientBoosting': GBpreds, 'kNN': knnpreds, 'MultiLayerPerceptron': Mlppreds, 'LightGBM': Lightpred, 'XGBoost': Xgbpred}
# MutluRES = pd.DataFrame(y)
# MutluRES.to_csv('MutluRES.csv', index = True)


