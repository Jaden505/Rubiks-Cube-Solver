import ReplayBuffer as rb

from numpy import array

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
        state = array([state])  # Add batch dimension
        prediction = self.model.predict(state).tolist()[0]

        face_index = prediction.index(max(prediction[0:5]))  # Get face with the highest probability
        direction = "clockwise" if prediction[6] > 0.5 else "counterclockwise"  # Get direction with the highest probability

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
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, batch, epochs=100, batch_size=90):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        batch.sort(key=lambda b: b["reward"], reverse=True)  # Sort by reward
        # batch = batch[:int(len(batch) * 0.2)]  # Keep only the top 20% of the batch

        print([b["reward"] for b in batch])

        x = [b["state"] for b in batch]

        y = []
        for b in batch:
            action = b["action"]
            temp = [0, 0, 0, 0, 0, 0, 0]

            temp[action[0]] = 1  # Set face index to 1
            temp[-1] = 1 if action[1] == "clockwise" else 0  # Set direction to 1 if clockwise, 0 if counterclockwise

            y.append(temp)

        hist = self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        # self.model.save('model.h5')

        return hist.history['loss'][-1]
