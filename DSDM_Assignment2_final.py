#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import RFECV
from xgboost.sklearn import XGBClassifier

data = pd.ExcelFile('Data.xlsx')
plants = pd.read_excel(data, 'plants')
flight = pd.read_excel(data, 'flight dates')
planting = pd.read_excel(data, 'planting')
weather = pd.read_excel(data, 'weather')


# In[2]:


# Renaming the plants data columns
plants = plants.rename(columns = {'Batch Number': 'batch_number', 'Plant Date': 'plant_date', 'Class': 'class', 
                            'Fresh Weight (g)': 'fresh_weight', 'Head Weight (g)': 'head_weight', 
                            'Radial Diameter (mm)': 'radial_diameter', 'Polar Diameter (mm)': 'polar_diameter', 
                            'Diameter Ratio': 'diameter_ratio', 'Leaves': 'leaves', 'Density (kg/L)': 'density',
                            'Leaf Area (cm^2)': 'leaf_area', 'Square ID': 'square_id', 
                            'Check Date': 'check_date', 'Flight Date': 'flight_date', 'Remove': 'remove'})
plants.describe()


# In[3]:


# Dropping the wrong garbage data after row 1822 
planting = planting.iloc[0:1821, :]
planting = planting.drop(columns = ['Column2', 'Column3', 'Column1', 'Column4'])
planting


# In[4]:


# Removing all the non-null values from the 'Remove column'
plants = plants[plants['remove'].isnull()]

# Dropping the remove column from the dataset
plants = plants.drop(columns = ['remove'])

# Dropping the leaves column
plants = plants.drop(columns = ['leaves'])


# In[5]:


# the number of NaN values in the plants plant_date
plants['plant_date'].isna().sum()


# In[6]:


# rename the flights data columns
flight = flight.rename(columns = {'Batch Number': 'batch_number', 'Flight Date': 'flight_date'})


# In[7]:


# Merging the plants and flight data on 'batch_number'
df_merge = pd.merge(plants, flight, how = 'left', on = 'batch_number')
dd1 = df_merge.loc[: , df_merge.columns != 'flight_date_x']
dd2 = df_merge.drop('flight_date_y', axis = 1)

dd1 = dd1.rename(columns = {'flight_date_y': 'flight_date'})
dd2 = dd2.rename(columns = {'flight_date_x': 'flight_date'})

dd1.update(dd2)
df_merge = dd1


# In[8]:


### Dropping the NaN values of the flight_date, head_weight, radial_diameter, polar_diameter
plant = df_merge.dropna(subset = ['flight_date', 'head_weight', 'radial_diameter', 'polar_diameter'])


# In[9]:


plant = plant.copy()


# In[10]:


plant


# In[11]:


### dropping the rows with Null values in plant_date
plant.dropna(subset = ['plant_date'], inplace = True)


# In[12]:


### Making a new variable 'flight_time' which tells the number of days from the 'plant_date'
plant['flight_time'] = plant['flight_date'] - plant['plant_date']
plant['flight_time'] = plant['flight_time'].astype('timedelta64[D]')


# In[13]:


plant['check_time'] = plant['check_date'] - plant['plant_date']
plant['check_time'] = plant['check_time'].astype('timedelta64[D]')


# In[14]:


plant['check_flight_time'] = plant['check_date'] - plant['flight_date']
plant['check_flight_time'] = plant['check_flight_time'].astype('timedelta64[D]')


# In[15]:


### dropping all the Null values in the plants
plant.dropna(inplace=True)


# In[16]:


### changing the plant-date and check-date to date time format
plant['plant_date']= pd.to_datetime(plant['plant_date'])
plant['check_date']= pd.to_datetime(plant['check_date'])


# In[17]:


### renaming the columns of the weather data
weather = weather.rename(columns = {'Unnamed: 0': 'weather_date', 'Solar Radiation [avg]': 'solar_radiation', 
                            'Precipitation [sum]': 'precipitation', 'Wind Speed [avg]': 'wind_speed_avg',
                            'Wind Speed [max]': 'wind_speed_max', 'Battery Voltage [last]': 'battery_voltage',
                           'Leaf Wetness [time]': 'leaf_wetness', 'Air Temperature [avg]': 'air_temp_avg',
                           'Air Temperature [max]': 'air_temp_max', 'Air Temperature [min]': 'air_temp_min',
                           'Relative Humidity [avg]': 'relative_humidity', 'Dew Point [avg]': 'dew_point_avg',
                           'Dew Point [min]': 'dew_point_min', 'ET0 [result]': 'eto_result'})


# In[18]:


###  dropping the duplpicates in the weather dataset
weather = weather.drop_duplicates(subset = ['weather_date'])


# In[19]:


### changing the weather-date to date time format
weather['weather_date']= pd.to_datetime(weather['weather_date'])


# In[20]:



for x,(i, j) in enumerate(zip(plant.plant_date, plant.check_date)):
    df_subset = weather[(weather['weather_date']>i) & (weather['weather_date']< j)]
    plant.at[x, 'avg_precipitation'] = (df_subset['precipitation'].mean())
    plant.at[x, 'std_precipitation'] = (df_subset['precipitation'].std())
    plant.at[x, 'avg_solar_rad'] = df_subset['solar_radiation'].mean()
    plant.at[x, 'std_solar_rad'] = df_subset['solar_radiation'].std()
    plant.at[x, 'avg_wind_speed'] = df_subset['wind_speed_avg'].mean()
    plant.at[x, 'std_wind_speed'] = df_subset['wind_speed_avg'].std()
    plant.at[x, 'avg_air_temp'] = df_subset['air_temp_avg'].mean()
    plant.at[x, 'std_air_temp'] = df_subset['air_temp_avg'].std()
    plant.at[x, 'avg_leaf_wetness'] = df_subset['leaf_wetness'].mean()
    plant.at[x, 'std_leaf_wetness'] = df_subset['leaf_wetness'].std()
    plant.at[x, 'avg_relative_humidity'] = df_subset['relative_humidity'].mean()
    plant.at[x, 'std_relative_humidity'] = df_subset['relative_humidity'].std()
    plant.at[x, 'avg_dew_point'] = df_subset['dew_point_avg'].mean()
    plant.at[x, 'std_dew_point'] = df_subset['dew_point_avg'].std()


# In[21]:


### dropping the rows with Null values again if any
plant = plant.dropna()


# In[22]:


plant = plant[['plant_date', 'flight_date', 'check_date','batch_number', 'class', 'density', 
               'leaf_area','square_id', 
               'flight_time', 'check_time', 'check_flight_time', 
               'avg_precipitation', 'std_precipitation', 'avg_solar_rad', 'std_solar_rad',
               'avg_wind_speed','std_wind_speed', 'avg_air_temp', 'std_air_temp', 
               'avg_leaf_wetness', 'std_leaf_wetness', 'avg_relative_humidity','std_relative_humidity',
               'avg_dew_point','std_dew_point' ,'fresh_weight', 'diameter_ratio',
               'head_weight', 'radial_diameter', 'polar_diameter']]


# In[23]:


plant


# In[97]:


### Exploratory Data Analysis

### Plant data analysis
plant_data = plant[['batch_number', 'class', 
                     'density', 'leaf_area','square_id', 'flight_time', 'check_time' ,'fresh_weight', 
                    'diameter_ratio', 'head_weight', 'radial_diameter', 'polar_diameter']]

plant_data.hist(figsize = (16,10))
plt.savefig("plant_hist.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[98]:


### plant_data heatmap
fig = plt.figure(figsize = (10,10))
sns.heatmap(plant_data.corr(), vmax = 0.6, square = True)
plt.savefig("plant_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[106]:


sns.jointplot(x = "radial_diameter",y = "polar_diameter", data=plant, hue="class");
plt.savefig("radial_polar.pdf", format="pdf", bbox_inches="tight")
plt.suptitle("Joint plot between Fresh Weight and Head Weight", y = 0)
plt.show()


# In[107]:


sns.jointplot(x = "fresh_weight", y = "head_weight", data=plant
              , hue="class");
plt.suptitle("Joint plot between Fresh Weight and Head Weight", y = 0)
plt.savefig("fresh_weight_head_weight.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[108]:


sns.scatterplot(data = plant, x="check_time", y="density", hue="class")
plt.title('Scatterplot between check_time - density',loc='center' ,y=-0.3)
plt.xlabel('check_time')
plt.ylabel('density')
plt.savefig("check_time_density.pdf", format="pdf", bbox_inches="tight")

plt.show()


# In[109]:


sns.pairplot(plant[['batch_number', 'class', 'flight_time' ,
                     'head_weight', 'radial_diameter', 'polar_diameter']])  

plt.savefig("plant_pairplot.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[110]:


### weather data analysis

weather.hist(figsize = (16,10))
plt.savefig("weather_histplot.pdf", format="pdf", bbox_inches="tight")

plt.show()


# In[111]:


fig = plt.figure(figsize = (10,10))
sns.heatmap(weather.corr(), vmax = .8, square = True)
plt.savefig("weather_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[37]:


X = plant.iloc[:, 3:-5]
y = plant.iloc[:, -3:]


# In[43]:


X = X.to_numpy()
y = y.to_numpy()


# In[45]:


### detection of Outliers
outliers = LocalOutlierFactor()
out = outliers.fit_predict(X)
# masking out by selecting all rows that are not outliers
mask = out != -1
X, y = X[mask, :], y[mask]
print(X.shape, y.shape)


# In[47]:


# Split the data into train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[127]:


X_plant = plant.iloc[:, 3:11]
y_plant = plant.iloc[:, -3:]
X_weather = plant.iloc[:, 11:25]
y_weather = plant.iloc[:, -3:]


# In[128]:


X_plant_train, X_plant_test, y_plant_train, y_plant_test = train_test_split(X_plant, y_plant, test_size=0.33, 
                                                                            random_state=42)
X_weather_train, X_weather_test, y_weather_train, y_weather_test = train_test_split(X_weather, y_weather, 
                                                                                    test_size=0.33, random_state=42)


# In[132]:


# Model 1 : Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('model score:' ,r2_score(y_test, y_pred, multioutput='variance_weighted'))


# In[129]:


# Model 1.1 : Linear Regression using just plants data
model = LinearRegression()
model.fit(X_plant_train, y_plant_train)
y_pred = model.predict(X_plant_test)
print('model score:' ,r2_score(y_plant_test, y_pred, multioutput='variance_weighted'))


# In[130]:


# Model 1.2 : Linear Regression using just weather data
model = LinearRegression()
model.fit(X_weather_train, y_weather_train)
y_pred = model.predict(X_weather_test)
print('model score:' ,r2_score(y_weather_test, y_pred, multioutput='variance_weighted'))


# In[135]:


# Model 2 : Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('model score:' ,r2_score(y_test, y_pred, multioutput='variance_weighted'))


# In[137]:


feat_importances = pd.Series(model.feature_importances_, index=plant.iloc[:, 3:-5].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.savefig("feature_imp_all.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[138]:


# Model 2.1 : Random Forest using just plant data
model = RandomForestRegressor()
model.fit(X_plant_train, y_plant_train)
y_pred = model.predict(X_plant_test)
print('model score:' ,r2_score(y_plant_test, y_pred, multioutput='variance_weighted'))


# In[140]:


feat_importances = pd.Series(model.feature_importances_, index=plant.iloc[:, 3:11].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.savefig("feature_imp_plant.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[141]:


# Model 2.2 : Random Forest using just weather data
model = RandomForestRegressor()
model.fit(X_weather_train, y_weather_train)
y_pred = model.predict(X_weather_test)
print('model score:' ,r2_score(y_weather_test, y_pred, multioutput='variance_weighted'))


# In[142]:


feat_importances = pd.Series(model.feature_importances_, index=plant.iloc[:, 11:25].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.savefig("feature_imp_weather.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[94]:


# list(plant.iloc[:, 3:-5].columns.values)


# In[ ]:





# In[93]:


# Model 2 : Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('model score:' ,r2_score(y_test, y_pred, multioutput='variance_weighted'))


# In[ ]:


# 'batch_number' ,'density' , 'leaf_area' , 'check_time', 'std_precipitation', 'avg_solar_rad',
#  'std_solar_rad', 'std_air_temp', 'avg_relative_humidity', 'std_relative_humidity', 'avg_dew_point', 
#     'std_dew_point'


# In[91]:


# Model 3 : Gradient Boosting

reg = MultiOutputRegressor(GradientBoostingRegressor())
reg.fit(X_train, y_train)
reg.score(X_test, y_test)


# In[ ]:




