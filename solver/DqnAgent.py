import ReplayBuffer as rb

from numpy import array, random, copy, amax
from random import random as rand_int

from keras import Sequential
from keras.layers import *
from keras.models import clone_model
from keras.metrics import AUC


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
        self.discount = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate

    def policy(self, state, model, train=True):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        # random_action = random.uniform(0, 1) < self.epsilon
        # if random_action:
        #     return self.epsilon_greedy_policy(train)

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
        return random action based formatted on training data or test data based on parameter
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
        The output is split into 2 branches one for the face and one for the direction
        """
        self.model.add(InputLayer(input_shape=(6, 3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(32, activation='elu'))

        # Define the output branches
        face_output = Dense(6, activation='softmax', name='face_output')
        direction_output = Dense(2, activation='softmax', name='direction_output')

        # Modify the output layer of the existing model to have two branches
        self.model.add(face_output)
        self.model.add(direction_output)

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
        current_q = face_current_q + direction_current_q

        target_q = copy(current_q)
        face_next_q, direction_next_q = self.model.predict(next_state_batch)  # Get q-values of state
        next_q = face_next_q + direction_next_q

        max_next_q = amax(next_q, axis=1)

        print(target_q)

        for i in range(state_batch.shape[0]):
            # If state of cube is solved set target q-value reward otherwise add maximum next state q-value
            face = int(action_batch[i][1])
            reward = reward_batch[i] if done_batch[i] \
                else reward_batch[i] + (self.discount * max_next_q[i])

            target_q[i][face] = reward  # Reward rotation face
            target_q[i][-1] = reward  # Reward rotation direction

        result = self.model.fit(x=state_batch, y=target_q, epochs=3)

        return result.history['loss']

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(self.model.get_weights())