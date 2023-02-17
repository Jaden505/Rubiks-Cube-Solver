import ReplayBuffer as rb

from numpy import array, random
from random import random as rand_int

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
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    def policy(self, state, model, train=True):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        random_action = random.uniform(0, 1) < self.epsilon
        if random_action:
            return self.epsilon_greedy_policy(train)

        normalised = [[[node / 6 for node in row] for row in face] for face in state]
        state = array([normalised])  # Add batch dimension
        prediction = model.predict(state).tolist()[0]

        if not train:  # If used for parameters to turn cube (not training)
            face_index = prediction.index(max(prediction[0:5]))  # Get face with the highest probability
            direction = "clockwise" if prediction[6] > 0.5 else "counterclockwise"  # Get direction with the highest probability
            return face_index, direction

        return prediction

    def epsilon_greedy_policy(self, train):
        """
        Epsilon greedy policy for exploration
        """
        if not train:
            face_index = random.randint(0, 5)
            direction = "clockwise" if random.random() < 0.5 else "counterclockwise"
            return face_index, direction

        else:
            empty_prediction = [0, 0, 0, 0, 0, 0, 0]
            empty_prediction[random.randint(0, 6)] = rand_int()
            return empty_prediction

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 6x3x3 cube
        The output is a 7x1 vector
        6 possible actions and 1 for the rotation direction (clockwise or counterclockwise)
        """
        self.model.add(InputLayer(input_shape=(6, 3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(32, activation='elu'))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    def train(self, batch):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        x, y = [], []

        for index, row in batch:
            state, next_state, reward, action, done = row["state"], row["next_state"], row["reward"], row["action"], row["done"]

            target_prediction = self.policy(state, self.target_model, False)
            future_target_prediction = self.policy(next_state, self.target_model)

            empty_prediction = [0, 0, 0, 0, 0, 0, 0]
            empty_prediction[-1] = 1 if action[1] == "clockwise" else 0

            if done:
                empty_prediction[target_prediction[0]] = 1
            else:
                future_reward = self.gamma * (max(future_target_prediction[:-1]) * 54)
                empty_prediction[target_prediction[0]] = (reward + future_reward) / 105  # Normalize

            x.append(state)
            y.append(empty_prediction)

            self.update_target_model()

        self.model.fit(x=x, y=y, epochs=1, batch_size=self.batch_size, verbose=1)

        # Decrease epsilon over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())