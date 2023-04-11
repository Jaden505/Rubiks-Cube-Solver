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

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 6x3x3 cube
        The output is split into 2 branches one for the face and one for the direction
        """
        input_layer = Input(shape=(54, 6))

        # Define the model layers
        x = Flatten()(input_layer)
        x = Dense(32, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)

        # Define the output branches
        face_output = Dense(6, activation='sigmoid', name='face_output')(x)
        direction_output = Dense(2, activation='sigmoid', name='direction_output')(x)

        self.model = Model(inputs=input_layer, outputs=[face_output, direction_output])

        # Define the loss function for each output
        losses = {
            'face_output': 'categorical_crossentropy',
            'direction_output': 'categorical_crossentropy'
        }

        # Compile the model with the optimizer, loss function, and metrics
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=losses, metrics=['mse'])

    @staticmethod
    def policy(state, model):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        @param state: The current state of the Rubik's cube
        @param model: The model to use to predict the action
        @return: The action to take
        """
        face_prediction, direction_prediction = model.predict(np.array([state]), verbose=0)
        face_prediction, direction_prediction = np.squeeze(face_prediction).tolist(), np.squeeze(
            direction_prediction).tolist()

        face_index = face_prediction.index(max(face_prediction))
        direction = 1 if direction_prediction[0] > direction_prediction[1] else 0

        return face_index, direction

    def train(self, batch):
        """
        Trains the model using the batch of data
        """
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = list(batch.values())
        state_batch = np.array([DqnAgent.one_hot_encode(state) for state in state_batch])
        next_state_batch = np.array([DqnAgent.one_hot_encode(state) for state in next_state_batch])

        faces_q, directions_q, next_target_q_faces, next_target_q_directions = \
            self.get_q_values(state_batch, next_state_batch)

        max_next_q_faces, max_next_q_directions = \
            np.amax(next_target_q_faces, axis=1), np.amax(next_target_q_directions, axis=1)

        for i in range(state_batch.shape[0]):
            face, direction = DqnAgent.policy(state_batch[i], self.model)

            reward_face = (reward_batch[i] + (0.95 * max_next_q_faces[i])) / 2.95
            reward_direction = (reward_batch[i] + (0.95 * max_next_q_directions[i])) / 2.95

            faces_q[i][face] = reward_face  # Reward rotation face
            directions_q[i][direction] = reward_direction  # Reward rotation direction

        self.model.fit(x=state_batch, y=[faces_q, directions_q])

    @staticmethod
    def one_hot_encode(state):
        # One hot encode the state
        flat_state = np.array(state).flatten()
        one_hot_identity = np.eye(6)
        return one_hot_identity[flat_state.astype(int)]

    def get_q_values(self, state_batch, next_state_batch):
        face_target_q, direction_target_q = self.model.predict(state_batch)  # Get q-values of state
        face_next_q, direction_next_q = self.target_model.predict(next_state_batch)  # Get q-values of state

        return face_target_q, direction_target_q, face_next_q, direction_next_q

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy_policy(self, state):
        """
        Epsilon greedy policy for exploration
        return random action based formatted on training data or test data based on parameter
        """
        if rand() < self.epsilon:
            face = np.random.randint(0, 5)
            direction = round(rand())
            return face, direction

        return DqnAgent.policy(state, self.model)
