
#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures



veri = pd.read_csv("2016dolaralis.csv")


x = veri["Gun"]
y = veri["Fiyat"]

x = x.reshape(251,1)
y= y.reshape(251,1)

plt.scatter(x,y)
plt.show()

#Lineer Reg.
tahminlineer = LinearRegression()
tahminlineer.fit(x,y)
tahminlineer.predict(x)

plt.plot(x,tahminlineer.predict(x),c="red")

#Polinom Reg.

tahminpolinom = PolynomialFeatures(degree=3)
Xyeni = tahminpolinom.fit_transform(x)

polinommodel = LinearRegression()
polinommodel.fit(Xyeni,y)
polinommodel.predict(Xyeni)

plt.plot(x,polinommodel.predict(Xyeni))

plt.show()

hatakaresilineer = 0
hatakaresipolinom = 0

for i in range(len(Xyeni)):
    hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2

for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(tahminlineer.predict(x)[i]))**2



"""
hatakaresipolinom = 0
    
for a in range(150):

    tahminpolinom = PolynomialFeatures(degree=a+1)
    Xyeni = tahminpolinom.fit_transform(x)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni,y)
    polinommodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2
    print(a+1,"inci dereceden fonksiyonda hata,", hatakaresipolinom)

    hatakaresipolinom = 0
  

"""








tahminpolinom8 = PolynomialFeatures(degree=8)
Xyeni = tahminpolinom8.fit_transform(x)

polinommodel8 = LinearRegression()
polinommodel8.fit(Xyeni,y)
polinommodel8.predict(Xyeni)

plt.plot(x,polinommodel8.predict(Xyeni))

plt.show()













print((float(y[201])-float(polinommodel8.predict(Xyeni)[201])))
