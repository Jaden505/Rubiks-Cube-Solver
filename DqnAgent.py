import ReplayBuffer as rb

import math
import numpy as np

from keras import Sequential
from keras.layers import *


class DqnAgent:
    """
      DQN Agent: Train using the DQN algorithm
      and will be receiving random states from the Rubik's cube
      """
    def __init__(self):
        self.replay_buffer = rb.ReplayBuffer()
        self.model = Sequential()
        self.create_model()

    def policy(self, state):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        state = np.array([state])  # Add batch dimension
        prediction = self.model.predict(state).tolist()[0]

        face_index = prediction.index(max(prediction[0:5]))  # Get face with the highest probability
        direction = "clockwise" if prediction[6] > 0.5 else "counterclockwise"  # Get direction with the highest probability

        print(face_index, direction)

        return face_index, direction

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 6x3x3 cube
        The output is a 7x1 vector
        6 possible actions and 1 for the rotation direction (clockwise or counterclockwise)
        """
        self.model.add(InputLayer(input_shape=(6, 3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, batch, epochs=10, batch_size=32):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        x = batch["state"]
        y = batch["action"]

        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        self.model.save('model.h5')
