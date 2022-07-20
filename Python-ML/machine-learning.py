#importing the libraries
from timeit import default_timer as timer
start = timer()

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from sklearn.datasets import load_iris

engine = create_engine('postgresql://username:password@postgresql-db-name.postgres.database.azure.com:5432/database')
df = pd.read_sql("iris_data", engine)

#exclude results for later checks
x = df.iloc[:, [1, 2, 3, 4]].values

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
  
df["cluster"] = y_kmeans
df.loc[df["cluster"] == 1, "cluster"] = "setosa"
df.loc[df["cluster"] == 0, "cluster"] = "versicolor"
df.loc[df["cluster"] == 2, "cluster"] = "virginica"
    
#upload log entry
end = timer()
date = datetime.now()
time_taken = end - start

#dd/mm/YY H:M:S
dt_string = date.strftime("%d/%m/%Y %H:%M:%S")
log = {'Name':['DL - Azure Function'],'Timestamp':[dt_string], 'Zeit gebraucht':[time_taken]}
df_log = pd.DataFrame(log)

#importing the Iris dataset with pandas
df_log.to_sql('log', con=engine, if_exists='append')
df.to_sql('iris_data_test', con=engine, if_exists='replace')