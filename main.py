import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb

from numpy import random

class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()
        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.STEPS = 50
        self.DATA_SIZE = 500

    def train_model(self):
        for i in range(self.STEPS):
            self.get_train_data()

            batch = self.buffer.sample_gameplay_batch(int(self.DATA_SIZE * 0.4))
            self.agent.train(batch)

            if i % 5 == 0:
                self.agent.update_target_model()

        self.agent.model.save("model.h5")

    def get_train_data(self):
        self.cube.scramble()
        state = self.cube.get_cube_state()

        for i in range(self.DATA_SIZE):
            action = self.agent.policy(state, self.agent.target_model, train=False)

            # if step < 1:
            #     action = random.randint(0, 6), "clockwise" if random.random() < 0.5 else "counterclockwise"

            next_state, reward, done = self.cube.step(action)

            print("Action: ", action)
            print("Reward: ", reward)

            self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = next_state

            if done:
                break


if __name__ == "__main__":
    m = Main()
    m.train_model()
