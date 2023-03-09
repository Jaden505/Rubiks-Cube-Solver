import DqnAgent as da

from keras.models import clone_model
from numpy import copy, amax, random


class AgentHelper(da.DqnAgent):
    def __init__(self):
        super().__init__()

        self.target_model = clone_model(self.model)
        self.update_target_model()

    def get_q_values(self, state_batch, next_state_batch):
        face_current_q, direction_current_q = super().model.predict(state_batch)  # Get q-values of state
        face_target_q, direction_target_q = copy(face_current_q), copy(
            direction_current_q)  # Copy q-values to target q-values
        face_next_q, direction_next_q = self.target_model.predict(next_state_batch)  # Get q-values of state

        max_next_q_face = amax(face_next_q, axis=1)
        max_next_q_direction = amax(direction_next_q, axis=1)

        return face_target_q, direction_target_q, max_next_q_face, max_next_q_direction

    def update_target_model(self):
        """
        Updates the target model with the current model's weights
        """
        self.target_model.set_weights(super().model.get_weights())

    def epsilon_greedy_policy(self, state):
        """
        Epsilon greedy policy for exploration
        return random action based formatted on training data or test data based on parameter
        """
        if random.uniform(0, 1) < super().epsilon:
            face = random.randint(0, 5)
            direction = random.randint(0, 1)
            return face, direction

        return super().policy(state, super().model)
