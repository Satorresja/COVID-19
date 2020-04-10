from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import pylab as pl
import numpy as np
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import datetime as dt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
df = pd.read_csv("../input/Casos1.csv")
print(df.head(10))

Man=df["Departamento o Distrito"]=="Caldas"
Mani= df[Man]
cdf = Mani[['Fecha de diagnóstico','Departamento o Distrito','Edad','Sexo']]
fecha= cdf["Fecha de diagnóstico"]
dept=cdf["Departamento o Distrito"]
a = []
for i in range(0,len(dept)):
    a.append(i+1)

cdf["Counts"]=a
print(cdf.head(20))

gg=cdf[["Fecha de diagnóstico", "Counts"]]
gg=gg.rename(columns={"Fecha de diagnóstico":"dd", "Counts":"y"})
g=pd.to_datetime(gg["dd"],  dayfirst=True)
gg["ds"]=g
gg= gg[["ds", "y"]]


m = Prophet()
m.fit(gg)
future = m.make_future_dataframe(periods=20)
future.tail()
fcst = m.predict(future)
fig = m.plot(fcst,xlabel='Fecha',ylabel='Casos confirmados en Caldas')
print(fcst)
fig = m.plot_components(fcst)
f=fcst[["ds"]]
p=fcst[["yhat"]]
#print(f)


figure(figsize=(14, 10), dpi=300, facecolor="w", edgecolor="k")
#fig, ax = plt.subplots()
plt.plot(f[:19], a, 'bo', label='Datos reales',  markersize=12 ,color='black')
plt.plot(f[19:], p[19:], 'ro', label='Datos predecidos', markersize=12 ,color='orange')
plt.xlabel("Fecha", {"fontsize": 18})
plt.xticks(rotation=45)
plt.ylabel("Casos en Caldas", {"fontsize": 18})
plt.ylim(0,40)
plt.grid(True)
plt.title('Predicción de casos de Coronavirus en Caldas hasta el 28 de abril de 2020', {'fontsize': 20})
plt.legend(loc='best', fontsize=20)
plt.show()
plt.savefig("Predicción Caldas.pdf")