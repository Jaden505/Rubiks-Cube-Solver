from keras import Sequential
from keras.layers import *

class NN:
    def __init__(self):
        self.model = Sequential()

    def create_model(self):
        self.model.add(Dense(64, input_dim=54, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(54, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, x, y, epochs=10, batch_size=32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model.save('model.h5')

    def evaluate(self, x, y):
        self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)