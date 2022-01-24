from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from WorldWeatherPy import DetermineListOfAttributes
from WorldWeatherPy import HistoricalLocationWeather
from wwo_hist import retrieve_hist_data
import pandas as pd
import numpy as np

#%% Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#%% Defining RMSE function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#%% IMPORT EXCEL

location1 = ['38.05,35.46']
for iteration in range(len(location1)):
    frequency=1
    start_date = '16-FEB-2016'
    end_date = '16-JAN-2021'
    api_key = '6e3be25b74324173a8172756210704'
    location_list = [location1[iteration]]
    hist_weather_data = retrieve_hist_data(api_key,location_list,start_date,end_date,frequency,location_label = False,export_csv = True,store_df = True)

yeni1 = pd.read_csv('38.05,35.46.csv')

#%% Creating DateTime

start = pd.to_datetime('2016-02-16')
end = pd.to_datetime('2021-01-17')
rng = pd.date_range(start, end, freq='H')
len(rng)
rng = rng[0:-1]
len(rng)

# print(rng)


#%% Merge


Target = pd.read_excel('uzl-saatlik-bildirim - 2021-04-13T155331.315.xls', index_col=0)
Target = Target.reset_index('DateTime')
Target = Target.drop('DateTime', axis = 1)

Target['DateTime'] = rng
yeni1['DateTime'] = rng


df = pd.merge(yeni1, Target, on="DateTime", how="left")

# result = pd.read_excel('T10T90.xlsx', index_col=0)

# df = pd.merge(yeni1, result, on="DateTime", how="left")


# result = df['T50']
df = df.drop('UEVM (MWh)', axis = 1)
# df = df.drop('T90', axis = 1)
# df = df.drop('T10', axis = 1)

df = df.set_index('DateTime')

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
df = df.drop('date_time', axis = 1)

df = pd.merge(df, result, on="DateTime", how="left")

result = df['T50']
df = df.drop('T50', axis = 1)
# df = df.drop('T90', axis = 1)
# df = df.drop('T10', axis = 1)

# Target = df['Gen']
# df = df.drop('Gen', axis = 1)


# df['Sin'] = np.sin(pd.DatetimeIndex(df.index).hour)
# df['Cos'] = np.cos(pd.DatetimeIndex(df.index).hour)

# df['Sin_windspeedKmph'] = np.sin(df['windspeedKmph'])
# df['Cos_windspeedKmph'] = np.cos(df['windspeedKmph'])

# df['Sin_winddirDegree'] = np.sin(df['winddirDegree'])
# df['Cos_winddirDegree'] = np.cos(df['winddirDegree'])

# df['Week'] = pd.DatetimeIndex(df.index).dayofweek
# df['Hour'] = pd.DatetimeIndex(df.index).hour
df['Month'] = pd.DatetimeIndex(df.index).month
# df['WeekXMonth'] = df['Week'] * df['Month']
# df['Sin'] = np.sin(pd.DatetimeIndex(df.index).hour)
# df['Cos'] = np.cos(pd.DatetimeIndex(df.index).hour)

#%% ADDING NEW COLUMN

# df['LaggedDay1'] = np.nan
# df['LaggedDay1'].iloc[24:] = Target['Gen'][0:-24]

# df = df[24:]
# Target = Target[24:]
Target = Target.set_index('DateTime')
df = df.drop('Gen', axis = 1)

#%% Correlation Heatmap

ExtractDataCorrs = df.corr()
plt.figure(figsize = (12, 8))
# Heatmap of correlations
plt.show()
sns.heatmap(ExtractDataCorrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')

#%%
    
# for i in range(len(df)):
#     df['Direction'].iloc[i] = wind_diraction_to_angle(df['Direction'].iloc[i])
    
# df = df.replace({0 : 0.01})
# Target = Target.replace({0 : 0.01})

# df = df.astype(float)
# Target = Target.astype(float)

# df.to_csv('yenidataframe.csv', index = True)

# Target.to_csv('Target.csv', index = True)

#%% Feature Scaling    

# scaler = StandardScaler()
                                                               
# dfnew = sc_X.fit_transform(df.values)
# Targetnew = sc_y.fit_transform(Target.values.reshape(-1,1))


#%% SPLITTING TRAIN AND TEST SET

splitting = -24
X_train = df[0:splitting]
y_train = Target[0:splitting]

X_test = df[splitting:]
y_test = Target[splitting:]

#%% Initialize CatBoostRegressor

def CatBoost(X_train,y_train,X_test):
    grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

    model1 = GridSearchCV(estimator = CatBoostRegressor(), param_grid = grid, cv=5, n_jobs = -1, verbose = 2)
    model1.fit(X_train, y_train)
    Catpreds = model1.predict(X_test)
    return Catpreds

#%% Initialize AdaBoostRegressor

def AdaBoost(X_train,y_train,X_test):
    search_grid={'n_estimators': np.arange(100, 1000, 250), 'learning_rate':[0.001,0.01,0.1], 'random_state':[1]}
    ada = GridSearchCV(estimator = AdaBoostRegressor(), param_grid=search_grid, cv = 5,  n_jobs = -1, verbose = 2)
    ada.fit(X_train, y_train)
    Adapreds = ada.predict(X_test)
    return Adapreds

#%% Initialize MultilayerPerceptronRegressor

def MultiLayerPerceptron(X_train,y_train,X_test):
    param_list = {'solver': ['lbfgs', 'adam'], 'max_iter': np.arange(100, 200, 50), 'alpha': [0.0001] , 'hidden_layer_sizes': np.arange(10, 15), 'random_state':[8],'learning_rate': ['constant']}
    # param_list = {"hidden_layer_sizes": [1,50], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}
    mlp = GridSearchCV(estimator = MLPRegressor(), param_grid = param_list, cv=5, n_jobs = -1, verbose = 2)
    mlp.fit(X_train, y_train)
    Mlppreds = mlp.predict(X_test)
    return Mlppreds

#%% Initialize RandomForestRegressor

def RandomForest(X_train,y_train,X_test):
    param_grid = {'max_depth': [80, 90, 100], 'max_features': [2, 3], 'min_samples_leaf': [3, 4], 'min_samples_split': [8, 10], 'n_estimators': [100]}
    rf = GridSearchCV(estimator = RandomForestRegressor(), param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    rf.fit(X_train, y_train)
    # rf = RandomForestRegressor(n_estimators=100)
    rf = RandomForestRegressor(rf.best_params_['n_estimators'])
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    # Get predictions
    RFpreds = rf.predict(X_test)
    return RFpreds

#%% Initialize RidgeRegressor

def RidgeRegressor(X_train,y_train,X_test):
    params_Ridge = {'alpha': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100], "fit_intercept": [True, False], "solver": ['svd', 'lsqr']}
    Ridge_GS = GridSearchCV(estimator = Ridge(), param_grid=params_Ridge, cv = 5, n_jobs=-1, verbose = 2)
    Ridge_GS.fit(X_train, y_train)
    Ridge_GS.best_params_
    # Get predictions
    Ridgepreds = Ridge_GS.predict(X_test)
    return Ridgepreds

#%% Initialize LassoRegressor
def LassoRegressor(X_train,y_train,X_test):
    params_Lasso = {'alpha': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 1, 2, 3, 5, 8, 10, 20, 50, 100]}
    Lasso1 = GridSearchCV(estimator = Lasso(), param_grid = params_Lasso, n_jobs=-1, cv = 5, verbose = 2)
    Lasso1.fit(X_train, y_train)
    Lassopreds = Lasso1.predict(X_test)
    return Lassopreds

#%% Initialize GradientBoostingRegressor

def GradientBoosting(X_train,y_train,X_test):
    parameters = {'learning_rate': [0.01], 'subsample': [0.5, 0.2], 'n_estimators' : [100,500], 'max_depth': [4,6,8]}
    grid_GBR = GridSearchCV(estimator = GradientBoostingRegressor(), param_grid = parameters, verbose = 2)
    grid_GBR.fit(X_train, y_train)
    GBpreds = grid_GBR.predict(X_test)
    return GBpreds

#%% Initialize KNeighborsRegressor

def Knn(X_train,y_train,X_test):
    param_grid = {'n_neighbors': np.arange(1, 100, 2), 'weights': ['uniform', 'distance']}
    knn = KNeighborsRegressor(metric='euclidean')
    model = GridSearchCV(knn, param_grid, n_jobs=-1, cv=5, verbose=2)
    model.fit(X_train, y_train)
    model.best_params_
    # Get predictions
    knnpreds = model.predict(X_test)
    return knnpreds
#%% PREDICTION

Catpreds = CatBoost(X_train,y_train,X_test)
Adapreds = AdaBoost(X_train,y_train,X_test)
RFpreds = RandomForest(X_train,y_train,X_test)
Ridgepreds = RidgeRegressor(X_train,y_train,X_test)
Lassopreds = LassoRegressor(X_train,y_train,X_test)
GBpreds = GradientBoosting(X_train,y_train,X_test)
knnpreds = Knn(X_train,y_train,X_test)
Mlppreds = MultiLayerPerceptron(X_train,y_train,X_test)

r2_score(y_test, RFpreds)

#%% ERROR PREDICTION

RMSECatpreds = rmse(Catpreds, y_test.values)
RMSEAdapreds = rmse(Adapreds, y_test.values)
RMSERFpreds = rmse(RFpreds, y_test.values)
RMSERidgepreds = rmse(Ridgepreds, y_test.values)
RMSELassopreds = rmse(Lassopreds, y_test.values)
RMSEGBpreds = rmse(GBpreds, y_test.values)
RMSEknnpreds = rmse(knnpreds, y_test.values)
RMSEMlppreds = rmse(Mlppreds, y_test.values)

RMSET50 = rmse(result.tail(24).values, y_test.values)
#%% PLOT

x_ax = range(len(y_test))
plt.plot(x_ax, list(y_test.values), linewidth=1, label="original")
plt.plot(x_ax, list(Ridgepreds), linewidth=1.1, label="predicted")
plt.plot(x_ax, list(result.tail(24).values), linewidth=1.1, label="Meteo")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

#%% RMSE Plotting

x = ['CatBoost','Adaboost','RandomForest','Ridge','Lasso','GradientBoosting','kNN', 'MultiLayerPerceptron']
y = [RMSECatpreds, RMSEAdapreds, RMSERFpreds, RMSERidgepreds, RMSELassopreds, RMSEGBpreds, RMSEknnpreds, RMSEMlppreds]
newdf = pd.DataFrame({"Methods":x, "Results":y})
newdf_sorted = newdf.sort_values('Results',ascending=False)
plt.barh('Methods', 'Results',data=newdf_sorted)
plt.title('Error Prediction')
plt.show()



# substractionLasso = np.subtract(y_test, list(Mlppreds))
# substractionMeteo = np.subtract(y_test, list(result['P50'].tail(24)))
# substractionLasso = np.absolute(substractionLasso)
# substractionMeteo = np.absolute(substractionMeteo)
# for i in range(0,len(substractionMeteo)):
#     if substractionLasso.iloc[i] <= substractionMeteo.iloc[i]:
#         print('True')
#     else:
#         print('False')



# start = pd.to_datetime('2021-01-11')
# end = pd.to_datetime('2021-01-16')
# rng = pd.date_range(start, end, freq='3H')

# rng1 = rng[7:17]
# rng2 = rng1.append(rng[18:21])
# rng3 = rng2.append(rng[24:35])
# len(rng3)

# print(rng3)

# # initialize list of lists
# data = {'Predictions': list(NN), 'Meteologica': list(result['P50'].tail(24).values), 'Gen': list(y_test)}
  
# # Create the pandas DataFrame
# kups = pd.DataFrame(data)

# rng3 = pd.DataFrame(rng3).set_axis(['DateTime'], axis=1, inplace=False)

# kupsneuralnetwork = pd.concat([kups, rng3], axis=1)

# kupsneuralnetwork.to_excel('kupsneuralnetwork.xlsx', index = True)



