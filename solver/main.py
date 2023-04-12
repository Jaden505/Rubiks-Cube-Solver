from cube.helper_cube import CubeHelper
from solver.dqn_agent import DqnAgent
from solver.replay_buffer import ReplayBuffer

import random
import copy


class Main:
    def __init__(self):
        self.cube = CubeHelper()

        self.agent = DqnAgent()
        self.buffer = ReplayBuffer()

        self.STEPS = 20
        self.DATA_SIZE = 328

    def train_model(self):
        for step in range(self.STEPS):
            batch = self.buffer.sample_gameplay_batch(self.DATA_SIZE)
            self.agent.train(batch)

            if step % 2 == 0:
                self.agent.update_target_model()

        self.agent.model.save("../models/model2.h5")

    def get_train_data(self):
        state = copy.deepcopy(self.cube.get_cube_state())

        for i in range(5000):
            action = self.get_random_action()
            next_state, reward, done = self.cube.step(state, action)

            self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = copy.deepcopy(next_state)

        self.buffer.save()

    def get_random_action(self):
        return random.choice(self.cube.cube_rotations)


if __name__ == "__main__":
    m = Main()
    m.train_model()
    # m.get_train_data()