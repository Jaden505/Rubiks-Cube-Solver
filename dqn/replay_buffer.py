import pandas as pd
import ast, random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=4000):
        self.df = pd.DataFrame(columns=["state", "next_state", "q_state", "q_next_state",
                                        "reward", "action", "done", "td_error", "ohe_state"])

        # Cast columns to lists to prevents pandas from converting them to strings
        list_columns = ['state', 'next_state', 'q_state', 'q_next_state']
        for column in list_columns:
            self.df[column] = self.df[column].apply(lambda x: ast.literal_eval(x))

        self.max_size = max_size
        self.old_replays = self.df.copy()

    def add_to_buffer(self, state, next_state, q_state, q_next_state, reward, action, done, td_error, ohe_state):
        row = {"state": state, "next_state": next_state, "q_state": q_state, "q_next_state": q_next_state,
               "reward": reward, "action": action, "done": int(done), "td_error": td_error, "ohe_state": ohe_state}

        self.df = pd.concat([self.df, pd.DataFrame.from_records([row])])

        if len(self.df) > self.max_size:
            if len(self.old_replays) < 8000:
                oldest_row = self.df.iloc[0]
                self.old_replays = pd.concat([self.old_replays, pd.DataFrame.from_records([oldest_row])])
            else:
                self.old_replays = self.old_replays.iloc[1:]

            self.df = self.df.iloc[1:]

    def sample_gameplay_batch(self, step, size):
        if step < 20:  # Don't sample from buffer until it has at least 20 experiences to introduce randomness
            return self.df.sample(n=size, random_state=42).drop(['td_error', 'ohe_state'], axis=1).to_dict('list')

        if random.random() < 0.1 and len(self.old_replays) > size:  # Sample from old experiences 10% of the time
            probabilities = self.old_replays['td_error'] / self.old_replays['td_error'].sum()
            sampled_df = self.old_replays.sample(n=size, weights=probabilities)
            return sampled_df.drop(['td_error', 'ohe_state'], axis=1).to_dict('list')

        probabilities = self.df['td_error'] / self.df['td_error'].sum()
        sampled_df = self.df.sample(n=size, weights=probabilities)
        return sampled_df.drop(['td_error', 'ohe_state'], axis=1).to_dict('list')

    def clear_buffer(self):
        self.__init__()

    def update_td_errors(self, model, target_model, batch_size):
        """
        Update the TD errors of the most recent experiences
        """
        self.df = self.df.reset_index(drop=True)
        experiences = self.df.tail(batch_size)  # Get the most recent experiences

        ohe_states = np.vstack(experiences['ohe_state'])
        actions = experiences['action'].astype('int32')
        rewards = experiences['reward'].astype('float32')

        current_q_values = model.predict(np.array([ohe_states]), verbose=0)
        next_q_values = target_model.predict(np.array([ohe_states]), verbose=0)

        # Calculate the new TD errors
        td_errors = np.abs(
            rewards + 0.95 * np.max(next_q_values, axis=1) - current_q_values[np.arange(batch_size), actions])

        self.df.loc[experiences.index, 'td_error'] = td_errors




        if len(self.old_replays) > batch_size:
            self.old_replays = self.old_replays.reset_index(drop=True)
            experiences = self.old_replays.tail(batch_size)  # Get the most recent experiences

            ohe_states = np.vstack(experiences['ohe_state'])
            actions = experiences['action'].astype('int32')
            rewards = experiences['reward'].astype('float32')

            current_q_values = model.predict(np.array([ohe_states]), verbose=0)
            next_q_values = target_model.predict(np.array([ohe_states]), verbose=0)

            # Calculate the new TD errors
            td_errors = np.abs(
                rewards + 0.95 * np.max(next_q_values, axis=1) - current_q_values[np.arange(batch_size), actions])

            self.old_replays.loc[experiences.index, 'td_error'] = td_errors
