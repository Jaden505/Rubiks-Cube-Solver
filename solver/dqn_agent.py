import replay_buffer as rb

import numpy as np
from random import random as rand

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

        self.epsilon = 1.0  # exploration rate

        self.rotation_dict = {0: "U", 1: "U'", 2: "U2", 3: "D", 4: "D'", 5: "D2", 6: "L", 7: "L'", 8: "L2", 9: "R",
                              10: "R'", 11: "R2", 12: "F", 13: "F'", 14: "F2", 15: "B", 16: "B'", 17: "B2"}

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a one-hot encoded array of the Rubik's cube
        The output consists of 18 nodes each representing a possible rotation of the cube
        """
        input_layer = Input(shape=(54, 6))
        x = Flatten()(input_layer)
        x = Dense(32, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        output_layer = Dense(18, activation="sigmoid")(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['mse'])

    def policy(self, state, model, get_index=False):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        @param state: The current state of the Rubik's cube
        @param model: The model to use to predict the action
        @return: The action to take
        """
        prediction_array = model.predict(np.array([state]), verbose=0)
        prediction_index = np.argmax(prediction_array)

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
            reward = (reward_batch[i] + (0.95 * max_next_q[i])) / 2.95  # Divide by 2.95 to normalize the reward
            rotation = self.policy(state_batch[i], self.model, get_index=True)

            if reward > 0.7:
                print(reward)

            target_q[i][rotation] = reward

        self.model.fit(x=state_batch, y=target_q)

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
