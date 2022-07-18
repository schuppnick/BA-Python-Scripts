import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Data = pd.read_csv('kc_house_data.csv')
engine = create_engine('postgresql://schupp:Lachnummer-1@schupptest.postgres.database.azure.com:5432/test')
df = pd.read_sql("housing_prices", engine)
df.head(5).T

df.drop(["index"], inplace=True, axis=1)

df.info()
df.describe().transpose()

#let's break date to years, months
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

#data visualization house price vs months and years
fig = plt.figure(figsize=(16,5))
fig.add_subplot(1,2,1)
df.groupby('month').mean()['price'].plot()
fig.add_subplot(1,2,2)
df.groupby('year').mean()['price'].plot()

# check if there are any Null values
#df.isnull().sum()

# drop unnecessary columns
df = df.drop(['date', 'id', 'zipcode'],axis=1)

X = df.drop('price',axis =1).values
y = df['price'].values

#splitting Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

#standardization scaler - fit&transform on train, fit only on test
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))

# Creating a Neural Network Model
# having 19 neuron is based on the number of available features
model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='Adam',loss='mse')

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128, epochs=400)
model.summary()

loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

y_pred = model.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))

# Visualizing Our predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)

#Perfect predictions
plt.plot(y_test,y_test,'r')

"""
#visualizing residuals
fig = plt.figure(figsize=(10,5))
residuals = (y_test- y_pred)
sns.displot(residuals)
"""

#build dataset for reupload
df_end = pd.DataFrame(X_test_copy, columns = df.drop("price", axis=1).columns)
df_end["actual_price"] = y_test_copy
df_end["predicted_price"] = y_pred

#upload test-data with predictions
engine = create_engine('postgresql://schupp:Lachnummer-1@schupptest.postgres.database.azure.com:5432/test')
df_end.to_sql('housing_prices_test', con=engine, if_exists='replace')