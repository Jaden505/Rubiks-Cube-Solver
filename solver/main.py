from cube import RubiksCube as rc
from solver import DqnAgent as da, ReplayBuffer as rb

from random import randint


class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()

        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.buffer.load()

        self.STEPS = 30
        self.DATA_SIZE = 64

    def train_model(self):
        for step in range(self.STEPS):
            batch = self.buffer.sample_gameplay_batch(self.DATA_SIZE)
            self.agent.train(batch)

            if step % 5 == 0:
                self.agent.update_target_model()

        self.agent.model.save("../models/model.h5")

    def get_train_data(self):
        state = self.cube.get_cube_state()

        for i in range(5000):
            action = self.get_random_action()
            next_state, reward, done = self.cube.step(action)

            print("Action: ", action)
            print("Reward: ", reward)
            print("State: ", state)
            print("Next state: ", next_state)

            self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = next_state

            if done:
                print("Solved!")
                break

        self.buffer.save()

    def get_random_action(self):
        directions = ['clockwise', 'counterclockwise']
        return randint(0, 5), directions[randint(0, 1)]


if __name__ == "__main__":
    m = Main()
    m.train_model()
    # m.get_train_data()