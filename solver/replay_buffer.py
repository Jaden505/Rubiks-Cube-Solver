import pandas as pd
import ast

class ReplayBuffer:
    def __init__(self):
        self.df = pd.DataFrame(columns=["state", "next_state", "reward", "action", "done"])

        # Cast column type
        self.df['done'] = self.df['done'].astype('bool').tolist()

    def add_to_buffer(self, state, next_state, reward, action, done):
        """
        Stores a step of gameplay experience in the buffer for later training
        """
        row = {"state": str(state), "next_state": str(next_state), "reward": reward, "action": action, "done": done}
        self.df = pd.concat([self.df, pd.DataFrame.from_records([row])])

    def sample_gameplay_batch(self, size):
        """
        Samples a batch of gameplay experiences for training
        """
        # Cast columns to lists
        self.df['state'] = self.df['state'].map(ast.literal_eval)
        self.df['next_state'] = self.df['next_state'].map(ast.literal_eval)

        return self.df.sample(min(self.df.size, size), random_state=42, replace=False).to_dict('list')

    def clear_buffer(self):
        self.__init__()
