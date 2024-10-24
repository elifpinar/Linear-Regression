# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:22:09 2022

@author: SerkanSavas
"""
#empty: rastgele değerler atıyor 
#numpy bilimsel hesaplamalarda kullanılır
#pandas veri işlemlerinde kullanılır, veri analizi
# dizi.dtype dizinin ögelerinin typini döndürür
#df.head()  , df.tail() : verinin son 5i , df.info(): dataframe hakkında bilgi verir, df.lo|'col'| : sütun başlığını verip sütüna erişmeyi sağlar
#df.groupbY()
#scikit learn açıkk kaynaklı bir makine öğrenmesi kütüphanesi 
#matplotlib görselleştirme 

#bağımsız değişkenler veri setindeki sütunlar eğitim , dedneyim, input, tahmin edici, öznitelik , x vb , bağımlı değişken: maaş, çıktılar, yanıtlar,y 
# regresyonda bağımlı değişkenler genelde sürekli değiğşkenlerdir (iki ölçüm arasında sonsuz sayıda değer alabilir) , bağımsızlar ayrık değişklenlerdir (sınırlı sayıda sonuç alabilir kadın,erkek vb.)
# genellikle nitel değişkenler ayrık , nicel değişkenlerde sürekli değişkenlerdir
# küçük kareler ilkesi
# hata metrikleri kullanılır mse vs.
# toplam kare hata en aza indirilecek şekilde hesaplanır 
# çoklu, çok değişkenli regresyon iki veya daha fazla bağımsız değişkenin olduğu lineer regresyon
# çoklu regresyon katsayısı :R 0 ile 1 arasında bir katsayı. 1 e ne kadar yaklaşırsa ilişki o kadar kuvvetlidir. ev  fiyatı merkeze yaklaştıkça fiystı artar.

#hata metriklerini araştır. 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

dataset = pd.read_csv('Kidem_ve_Maas_VeriSeti.csv')
dataset.head()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#Görselleştirme
#Veriler
plt.scatter(X_train, y_train, color = 'red')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()

#Veriler vs Tahmin
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = regressor.predict(X_train)
plt.scatter(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()

#Veriler vs Test Tahmin
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()


#Regresyon Çizgisi
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = regressor.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Kıdeme Göre Maaş Tahmini Regresyon Modeli')
plt.xlabel('Kıdem')
plt.ylabel('Maaş')
plt.show()

#Metrikler
r2_score(y_test, y_pred)
print("Ortalama Mutlak Hata: {} \nOrtalama Karesel Hata: {}".format(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test, y_pred)))
