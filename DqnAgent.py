import NeuralNet as nn
import ReplayBuffer as rb

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
        return self.nn.predict(state)

    def train(self, batch):
        """
        Trains the agent on a batch of data from the replay buffer.
        """
        loss = 0

        return loss
