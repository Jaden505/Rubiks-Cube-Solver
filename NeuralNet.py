from keras import Sequential
from keras.layers import *


class NN:
    def __init__(self):
        self.model = Sequential()

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 3x3x3 cube
        The output is a 7x1 vector
        6 possible actions and 1 for the rotation direction (clockwise or counterclockwise)
        """
        self.model.add(InputLayer(input_shape=(3, 3)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, x, y, epochs=10, batch_size=32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model.save('model.h5')

    def evaluate(self, x, y):
        self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)
