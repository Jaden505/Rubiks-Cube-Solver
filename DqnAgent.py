import ReplayBuffer as rb

from numpy import array

from keras import Sequential
from keras.layers import *
from keras.models import clone_model


class DqnAgent:
    """
      DQN Agent: Train using the DQN algorithm
      and will be receiving random states from the Rubik's cube
      """
    def __init__(self):
        self.replay_buffer = rb.ReplayBuffer()

        self.model = Sequential()
        self.create_model()

        self.target_model = clone_model(self.model)
        self.update_target_model()

        self.batch_size = 32
        self.gamma = 0.95   # discount rate
        self.learning_rate = 0.001

    def policy(self, state, train=True):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        state = array([state])  # Add batch dimension
        prediction = self.model.predict(state).tolist()[0]

        if not train:  # If used for parameters to turn cube (not training)
            face_index = prediction.index(max(prediction[0:5]))  # Get face with the highest probability
            direction = "clockwise" if prediction[6] > 0.5 else "counterclockwise"  # Get direction with the highest probability
            return face_index, direction

        return prediction

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

    def train(self, batch):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        for (state, next_state, reward, action, done) in batch:
            target_prediction = self.target_model.predict(array([state])).tolist()[0]
            future_target_prediction = self.target_model.predict(array([next_state])).tolist()[0]

            empty_prediction = [0, 0, 0, 0, 0, 0, 0]
            empty_prediction[-1] = 1 if action[1] == "clockwise" else 0

            future_reward = self.gamma * (max(future_target_prediction[:-1]) * 54)

            if done:
                empty_prediction[target_prediction[0]] = reward + future_reward

            self.model.fit(state, empty_prediction, epochs=1, batch_size=self.batch_size)

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())