from cube.helper_cube import CubeHelper
from solver.dqn_agent import DqnAgent
from solver.replay_buffer import ReplayBuffer

import copy


class Main:
    def __init__(self):
        self.cube = CubeHelper()
        self.agent = DqnAgent()
        self.buffer = ReplayBuffer()

        self.STEPS = 250
        self.BATCH_SIZE = 156
        self.TARGET_UPDATE = 16

    def train_model(self):
        for step in range(self.STEPS):
            self.get_train_data()
            batch = self.buffer.sample_gameplay_batch(self.BATCH_SIZE)
            self.agent.train(batch)

            if step % self.TARGET_UPDATE == 0:
                self.agent.update_target_model()

        self.agent.model.save("../models/model.h5")

    def get_train_data(self):
        self.cube.scramble()
        state = copy.deepcopy(self.cube.get_cube_state())

        for i in range(self.BATCH_SIZE):
            ohe_state = self.agent.one_hot_encode(state)
            action, q_state = self.agent.boltzmann_exploration(ohe_state, self.agent.model)
            next_state, reward, done = self.cube.step(self.agent.rotation_dict[action])
            _, q_next_state = self.agent.boltzmann_exploration(self.agent.one_hot_encode(next_state), self.agent.target_model)

            self.buffer.add_to_buffer(state, next_state, q_state, q_next_state, reward, action, done)
            state = copy.deepcopy(next_state)


if __name__ == "__main__":
    m = Main()
    m.train_model()
