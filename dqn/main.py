from cube.helper_cube import CubeHelper
from dqn.replay_buffer import ReplayBuffer
from dqn.dqn_agent import DqnAgent

from keras.models import load_model, clone_model
import copy


class Main:
    def __init__(self):
        self.cube = CubeHelper()
        self.buffer = ReplayBuffer()
        self.agent = DqnAgent()

        self.model_save_path = '../models/dqn/model.h5'
        model = load_model(self.model_save_path)
        self.agent.model = model
        self.agent.target_model = clone_model(model)

        self.STEPS = 2000
        self.BATCH_SIZE = 96
        self.TARGET_UPDATE = 5
        self.UPDATE_ALL_TD = 2

        self.solved_count = 0
        self.scramble_length = 2
        self.moves_since_scramble = 0

    def train_model(self):
        for step in range(self.STEPS):
            self.get_train_data()
            batch = self.buffer.sample_gameplay_batch(step, self.BATCH_SIZE)
            self.agent.train(batch)

            if step % self.UPDATE_ALL_TD == 0:
                self.buffer.update_td_errors(self.agent.model, self.agent.target_model, self.BATCH_SIZE)
            else:
                self.buffer.update_td_errors(self.agent.model, self.agent.target_model, self.buffer.df.shape[0])

            if step % self.TARGET_UPDATE == 0:
                self.agent.update_target_model()
                self.agent.model.save(self.model_save_path)

            if self.scramble_length >= 20:
                self.scramble_length = 2

        self.agent.model.save(self.model_save_path)

    def get_train_data(self):
        self.cube.reset()
        self.cube.scramble(self.scramble_length)
        state = copy.deepcopy(self.cube.get_cube_state())

        for i in range(self.BATCH_SIZE):
            self.moves_since_scramble += 1

            ohe_state = self.agent.one_hot_encode(state)
            action, q_state = self.agent.boltzmann_exploration(ohe_state, self.agent.model)
            next_state, reward, done = self.cube.step(self.agent.rotation_dict[action],
                                                      self.moves_since_scramble, self.scramble_length)
            _, q_next_state = self.agent.boltzmann_exploration(self.agent.one_hot_encode(next_state),
                                                               self.agent.target_model)
            td_error = abs(reward + (0.99 * max(q_next_state)) - q_state[action])

            self.buffer.add_to_buffer(state, next_state, q_state, q_next_state,
                                      reward, action, done, td_error, ohe_state)
            state = copy.deepcopy(next_state)

            if done:
                if self.solved_count >= 15:
                    self.scramble_length += 1
                    self.solved_count = 0
                else:
                    self.solved_count += 1

            if i % self.scramble_length == 0 or done:
                self.cube.reset()
                self.cube.scramble(self.scramble_length)
                state = copy.deepcopy(self.cube.get_cube_state())
                self.moves_since_scramble = 0


if __name__ == "__main__":
    m = Main()
    m.train_model()
