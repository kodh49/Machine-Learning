import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([1,2,3,4,5,6,7,8,9])
Y = np.array([11,22,33,44,53,66,77,87,95])

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(X, Y, batch_size=1, epochs=30, shuffle=False)

print(model.predict([9.5]))
plt.plot(X, model.predict(X), 'b', X, Y, 'k.')
plt.show()