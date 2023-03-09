import ReplayBuffer as rb

from numpy import array, squeeze

from keras.models import Model
from keras.layers import *


class DqnAgent:
    """
      DQN Agent: Train using the DQN algorithm
      and will be receiving random states from the Rubik's cube
      """

    def __init__(self):
        self.replay_buffer = rb.ReplayBuffer()

        self.model = None
        self.create_model()

        self.batch_size = 128
        self.discount = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 6x3x3 cube
        The output is split into 2 branches one for the face and one for the direction
        """
        input_layer = Input(shape=(6, 3, 3))

        # Define the model layers
        x = Flatten()(input_layer)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(32, activation='elu')(x)

        # Define the output branches
        face_output = Dense(6, activation='softmax', name='face_output')(x)
        direction_output = Dense(2, activation='softmax', name='direction_output')(x)

        self.model = Model(inputs=input_layer, outputs=[face_output, direction_output])

        # Define the loss function for each output
        losses = {
            'face_output': 'categorical_crossentropy',
            'direction_output': 'categorical_crossentropy'
        }

        # Define the loss weights for each output
        loss_weights = {
            'face_output': 1.0,
            'direction_output': 1.0
        }

        # Compile the model with the optimizer, loss function, and metrics
        self.model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    def policy(self, state, model, for_cube=True):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        normalised = [[[node / 6 for node in row] for row in face] for face in state]
        state = array([normalised])  # Add batch dimension
        face_prediction, direction_prediction = model.predict(state)
        face_prediction, direction_prediction = squeeze(face_prediction), squeeze(direction_prediction)

        if for_cube:  # If used for parameters to turn cube (not training)
            return self.action_to_parameters(face_prediction, direction_prediction)

        return face_prediction, direction_prediction

    def train(self, batch):
        """
        Trains the model using the batch of data
        """
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = list(batch.values())[1:]
        state_batch, next_state_batch = array(state_batch), array(next_state_batch)  # Set as np arrays to get q-values

        face_target_q, direction_target_q, max_next_q_face, max_next_q_direction = self.get_q_values(state_batch,
                                                                                                     next_state_batch)

        for i in range(state_batch.shape[0]):
            face, direction = self.epsilon_greedy_policy(state_batch[i])

            reward_face = reward_batch[i] if done_batch[i] \
                else reward_batch[i] + (self.discount * max_next_q_face[i])

            reward_direction = reward_batch[i] if done_batch[i] \
                else reward_batch[i] + (self.discount * max_next_q_direction[i])

            face_target_q[i][face] = reward_face  # Reward rotation face
            direction_target_q[i][direction] = reward_direction  # Reward rotation direction

        self.model.fit(x=state_batch, y=[face_target_q, direction_target_q], batch_size=self.batch_size, epochs=1,
                       verbose=1)
