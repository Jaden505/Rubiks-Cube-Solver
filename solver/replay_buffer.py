import pandas as pd
import ast, copy


class ReplayBuffer:
    def __init__(self):
        self.df = pd.DataFrame(columns=["state", "next_state", "q_state", "q_next_state", "reward", "action", "done"])

        self.df['state'] = self.df['state'].apply(lambda x: ast.literal_eval(x))
        self.df['next_state'] = self.df['next_state'].apply(lambda x: ast.literal_eval(x))
        self.df['q_state'] = self.df['q_state'].apply(lambda x: ast.literal_eval(x))
        self.df['q_next_state'] = self.df['q_next_state'].apply(lambda x: ast.literal_eval(x))

    def add_to_buffer(self, state, next_state, q_state, q_next_state, reward, action, done):
        row = {"state": state, "next_state": next_state, "q_state": q_state,
               "q_next_state": q_next_state, "reward": reward, "action": action, "done": done}

        self.df = pd.concat([self.df, pd.DataFrame.from_records([row])])
        self.df['done'] = self.df['done'].astype('int32').tolist()

    def sample_gameplay_batch(self, size):
        return self.df.sample(min(self.df.size, size), random_state=42, replace=False).to_dict('list')

    def clear_buffer(self):
        self.__init__()
