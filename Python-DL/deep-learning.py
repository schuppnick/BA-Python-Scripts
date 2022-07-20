from timeit import default_timer as timer
start = timer()

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

engine = create_engine('postgresql://schupp:Lachnummer-1@schupptest.postgres.database.azure.com:5432/test')
df = pd.read_sql("housing_prices", engine)

df.drop(["index"], inplace=True, axis=1)

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

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
            batch_size=128, epochs=100)
model.summary()

loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

y_pred = model.predict(X_test)

#build dataset for reupload
df_end = pd.DataFrame(X_test_copy, columns = df.drop("price", axis=1).columns)
df_end["actual_price"] = y_test_copy
df_end["predicted_price"] = y_pred

#upload log entry
end = timer()
date = datetime.now()
time_taken = end - start

#dd/mm/YY H:M:S
dt_string = date.strftime("%d/%m/%Y %H:%M:%S")
log = {'Name':['DL - Azure Function'],'Timestamp':[dt_string], 'Zeit gebraucht':[time_taken]}
df_log = pd.DataFrame(log)

df_log.to_sql('log', con=engine, if_exists='append')
#upload test-data with predictions
df_end.to_sql('housing_prices_test', con=engine, if_exists='replace')