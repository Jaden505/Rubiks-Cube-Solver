from cube.helper_cube import CubeHelper
from solver.dqn_agent import DqnAgent
from solver.replay_buffer import ReplayBuffer

import copy, random
import numpy as np


class Main:
    def __init__(self):
        self.cube = CubeHelper()

        self.agent = DqnAgent()
        self.buffer = ReplayBuffer()

        self.STEPS = 30
        self.BATCH_SIZE = 90

    def train_model(self):
        for step in range(1, self.STEPS):
            self.get_train_data()
            batch = self.buffer.sample_gameplay_batch(self.BATCH_SIZE)
            self.agent.train(batch)

            if step % 3 == 0:
                self.agent.update_target_model()

            self.buffer.clear_buffer()

        self.agent.model.save("../models/model.h5")

    def get_train_data(self):
        self.cube.scramble()
        state = copy.deepcopy(self.cube.get_cube_state())

        for i in range(self.BATCH_SIZE):
            if np.random.rand() < self.agent.epsilon:
                action = random.choice(self.cube.cube_rotations)
            else:
                action = self.agent.policy(self.agent.one_hot_encode(state), self.agent.model)

            next_state, reward, done = self.cube.step(action)
            self.buffer.add_to_buffer(state, next_state, reward, action, done)

            state = copy.deepcopy(next_state)

            if done:
                break


if __name__ == "__main__":
    m = Main()
    m.train_model()