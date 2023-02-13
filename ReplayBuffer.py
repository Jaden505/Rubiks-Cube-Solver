import random


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add_gameplay(self, state, next_state, reward, action, done):
        """
        Stores a step of gameplay experience in
        the buffer for later training
        """
        self.buffer.append({"state": state, "next_state": next_state, "reward": reward, "action": action, "done": done})

    def sample_gameplay_batch(self, batch_size):
        """
        Samples a batch of gameplay experiences
        for training purposes.
        """
        return random.sample(self.buffer, batch_size)
