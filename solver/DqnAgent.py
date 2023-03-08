import ReplayBuffer as rb

from numpy import array, squeeze, copy, amax

from keras.models import Model
from keras.layers import *
from keras.models import clone_model


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

        self.batch_size = 32
        self.discount = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate

    def policy(self, state, model, for_cube=True):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        normalised = [[[node / 6 for node in row] for row in face] for face in state]
        state = array([normalised])  # Add batch dimension
        face_prediction, direction_prediction = squeeze(model.predict(state))

        if for_cube:  # If used for parameters to turn cube (not training)
            return self.action_to_parameters(face_prediction, direction_prediction)

        return face_prediction, direction_prediction

    def action_to_parameters(self, face_prediction, direction_prediction):
        """
        Takes the action and returns the parameters to turn the cube
        """
        face_index = face_prediction.index(max(face_prediction))
        direction = "clockwise" if direction_prediction[0] > direction_prediction[1] else "counterclockwise"

        return face_index, direction

    def create_model(self):
        """
        Creates the model for the neural network
        The input is a 6x3x3 cube
        The output is split into 2 branches one for the face and one for the direction
        """
        input_layer = Input(shape=(6, 3, 3))

        # Define the model layers
        x = Flatten()(input_layer)
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

    def train(self, batch):
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = list(batch.values())[1:]
        state_batch, next_state_batch = array(state_batch), array(next_state_batch)  # Set as np arrays to get q-values

        face_current_q, direction_current_q = self.model.predict(state_batch)  # Get q-values of state
        face_target_q, direction_target_q = copy(face_current_q), copy(direction_current_q)  # Copy q-values to target q-values
        face_next_q, direction_next_q = self.target_model.predict(next_state_batch)  # Get q-values of state

        max_next_q_face = amax(face_next_q, axis=1)
        max_next_q_direction = amax(direction_next_q, axis=1)

        for i in range(state_batch.shape[0]):
            face = int(action_batch[i][1])
            direction = 1 if action_batch[i][0] == "clockwise" else 0

            reward_face = reward_batch[i] if done_batch[i] \
                else reward_batch[i] + (self.discount * max_next_q_face[i])

            reward_direction = reward_batch[i] if done_batch[i] \
                else reward_batch[i] + (self.discount * max_next_q_direction[i])

            target_q[i][face] = reward  # Reward rotation face
            target_q[i][-1] = reward  # Reward rotation direction

        result = self.model.fit(x=state_batch, y=target_q, epochs=3)

        return result.history['loss']

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())