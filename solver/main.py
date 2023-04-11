from cube import rubiks_cube as rc
from solver import dqn_agent as da, replay_buffer as rb

from random import randint
import copy


class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()

        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.buffer.load()

        self.STEPS = 100
        self.DATA_SIZE = 328

    def train_model(self):
        for step in range(self.STEPS):
            batch = self.buffer.sample_gameplay_batch(self.DATA_SIZE)
            self.agent.train(batch)

            if step % 3 == 0:
                self.agent.update_target_model()

        self.agent.model.save("../models/model.h5")

    def get_train_data(self):
        state = copy.deepcopy(self.cube.get_cube_state())

        for i in range(5000):
            action = self.get_random_action()
            next_state, reward, done = self.cube.step(state, action)

            self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = copy.deepcopy(next_state)

        self.buffer.save()

    def get_random_action(self):
        directions = ['clockwise', 'counterclockwise']
        return randint(0, 5), directions[randint(0, 1)]


if __name__ == "__main__":
    m = Main()
    m.train_model()
    # m.get_train_data()