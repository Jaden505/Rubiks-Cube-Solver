import replay_buffer as rb

import numpy as np
import random

from keras.models import clone_model
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam


class DqnAgent:
    """
      DQN Agent: Train using the DQN algorithm
      and will be receiving random states from the Rubik's cube
      """

    def __init__(self):
        self.replay_buffer = rb.ReplayBuffer()

        self.model = None
        self.create_model()

        self.target_model = clone_model(self.model)
        self.update_target_model()

        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01

        self.rotation_dict = {0: "U", 1: "U'", 2: "D", 3: "D'", 4: "L", 5: "L'",
                              6: "R", 7: "R'", 8: "F", 9: "F'", 10: "B", 11: "B'"}

        self.prev_pred = None

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a one-hot encoded array of the Rubik's cube
        The output consists of 18 nodes each representing a possible rotation of the cube
        """
        input_layer = Input(shape=(54, 6))
        x = Flatten()(input_layer)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(12, activation='softmax')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=0.005), loss='categorical_crossentropy', metrics=['mse'])

    def policy(self, state, model, get_index=False):
        """
        Takes a state from the Rubik's cube and returns an action that should be taken.
        """
        prediction_array = model.predict(np.array([state]), verbose=0)
        prediction_index = np.argmax(prediction_array)

        if prediction_index == self.prev_pred:
            print(prediction_index)

        self.prev_pred = prediction_index

        if get_index:
            return prediction_index

        return self.rotation_dict[prediction_index]

    def train(self, batch):
        """
        Trains the model using the batch of data
        """
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = list(batch.values())
        state_batch = np.array([DqnAgent.one_hot_encode(state) for state in state_batch])
        next_state_batch = np.array([DqnAgent.one_hot_encode(state) for state in next_state_batch])

        target_q, next_q = self.get_q_values(state_batch, next_state_batch)
        max_next_q = np.amax(next_q, axis=1)

        for i in range(state_batch.shape[0]):
            reward = (reward_batch[i] + (0.95 * max_next_q[i])) / 2.95

            if done_batch[i]:
                reward = 1  # If the cube is solved, give it a high reward

            target_q[i][action_batch[i]] = reward

        self.model.fit(x=state_batch, y=target_q)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  # Decay epsilon

    @staticmethod
    def one_hot_encode(state):
        # One hot encode the state
        flat_state = np.array(state).flatten()
        one_hot_identity = np.eye(6)
        return one_hot_identity[flat_state.astype(int)]

    def get_q_values(self, state_batch, next_state_batch):
        target_q = self.model.predict(state_batch)
        next_q = self.target_model.predict(next_state_batch)

        return target_q, next_q

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())
