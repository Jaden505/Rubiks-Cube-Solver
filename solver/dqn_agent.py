import copy

import replay_buffer as rb

import numpy as np

from keras.models import clone_model
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.initializers.initializers_v2 import GlorotUniform

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

        # Temperature for Boltzmann exploration: higher temperature means more exploration
        self.temp = 1.0
        self.temp_decay = 0.992
        self.temp_min = 0.01

        self.rotation_dict = {0: "U", 1: "U'", 2: "D", 3: "D'", 4: "L", 5: "L'",
                              6: "R", 7: "R'", 8: "F", 9: "F'", 10: "B", 11: "B'"}

        # self.prev_pred = None

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a one-hot encoded array of the Rubik's cube
        The output consists of 18 nodes each representing a possible rotation of the cube
        """
        input_layer = Input(shape=(54, 6))
        x = Flatten()(input_layer)
        x = Dense(512, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(512, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(256, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(256, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        output_layer = Dense(12, activation='softmax')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])

    def policy(self, state, model, get_index=False):
        """
        Takes a state from the Rubik's cube and returns an action that should be taken.
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
        state_batch, next_state_batch, q_state, q_next_state, reward_batch, action_batch, done_batch = list(batch.values())
        state_batch = np.array([DqnAgent.one_hot_encode(state) for state in state_batch])

        target_q = np.array(q_state)
        max_next_q = np.amax(q_next_state, axis=1)

        for i in range(state_batch.shape[0]):
            reward = (reward_batch[i] + (0.95 * max_next_q[i]))
            target_q[i][action_batch[i]] = reward

        self.model.fit(x=state_batch, y=target_q)
        self.temp = max(self.temp * self.temp_decay, self.temp_min)

    @staticmethod
    def one_hot_encode(state):
        """
        Input is a 6x3x3 array of the Rubik's cube
        Output is a one-hot encoded array of the Rubik's cube of shape 54x6
        """
        flat_state = np.array(state).flatten()
        one_hot_identity = np.eye(6)
        return one_hot_identity[flat_state.astype(int)]

    def boltzmann_exploration(self, ohe_state, model):
        """
        Input is a list of One-hot encoded values for each action
        Output is an action that should be taken and the Q-values for each action
        """
        q_state = model.predict(np.array([ohe_state]), verbose=0).tolist()[0]
        prob_values = self._softmax(np.array(q_state) / self.temp)
        return np.random.choice(len(prob_values), p=prob_values), q_state

    @staticmethod
    def _softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())