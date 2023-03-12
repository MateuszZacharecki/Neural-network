import numpy as np
import pandas as pd
#import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow._api.v1.keras import layers
#from keras import layers
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
#from sklearn import metrics
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    Trening = pd.read_csv("sc_project_training_data.csv")
    Test = pd.read_csv("sc_project_test_data.csv")

    df = pd.concat([Trening['GameType'], Test['GameType']])
    layer = tf.keras.layers.StringLookup()
    layer.adapt(df)
    Trening['TypeofGame'] = layer(Trening['GameType'])
    Test['TypeofGame'] = layer(Test['GameType'])

    Trening['Wynik'] = 0
    Trening.loc[Trening['Winner'] == 'A', 'Wynik'] = 1

    Test = Test.drop(columns=['GameType'])
    Wynik = Trening['Wynik']
    Trening = Trening.drop(columns=['GameID', 'Winner', 'GameType', 'Wynik'])

    scaler = MinMaxScaler()
    Trening_skalowany = scaler.fit_transform(Trening)
    Test_skalowany = scaler.fit_transform(Test)

    model = keras.Sequential()
    model.add(layers.Dense(32, input_shape=(158,), activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['AUC'])

    predictions = model.predict(Test_skalowany)

    f = open("projekt_sgd_1.txt", "w")
    for row in predictions:
        np.savetxt(f, row, fmt='%1.10f')
    f.close()

    dane_train, dane_test, wynik_train, wynik_test = train_test_split(Trening_skalowany, Wynik)
    pred = model.predict(dane_test)
    roc_auc_score(wynik_test, pred)
