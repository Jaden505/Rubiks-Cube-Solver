import NeuralNet as nn
import ReplayBuffer as rb

import math

class DqnAgent:
    """
      DQN Agent: Train using the DQN algorithm
      and will be receiving random states from the Rubik's cube
      """
    def __init__(self):
        self.nn = nn.NN()
        self.replay_buffer = rb.ReplayBuffer()

    def policy(self, state):
        """
        Takes a state from the Rubik's cube and returns
        an action that should be taken.
        """
        prediction = self.nn.predict(state)
        face = math.floor(max(prediction[0:5]))  # Get face with the highest reward
        direction = prediction[-1]

        return face, direction

    def train(self, batch):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        x = batch["state"]
        y = batch["action"]

        self.nn.train(x, y)
