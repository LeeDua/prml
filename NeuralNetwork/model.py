import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class HistoryRecorder(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

data = datasets.load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

batch_size = 16
max_epoches = 500

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])

history = HistoryRecorder()

model.fit(X_train,y_train,epochs=max_epoches,batch_size=batch_size,validation_data=(X_test,y_test),callbacks=[history])

loss_and_metrics = model.evaluate(X_test, y_test,batch_size=batch_size)

print(loss_and_metrics)

t = np.arange(max_epoches)
plt.plot(t,history.losses,history.accuracy)

plt.show()



