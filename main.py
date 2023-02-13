import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb

from numpy import random

class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()
        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.BATCH_SIZE = 350
        self.EPOCHS = 100
        self.DATA_SIZE = 400

    def train_model(self):
        for i in range(self.EPOCHS):
            self.get_train_data(i)
            batch = self.buffer.sample_gameplay_batch(350)
            loss = self.agent.train(batch)
            print("Loss: ", loss)


    def get_train_data(self, epoch):
        self.cube.scramble()
        state = self.cube.get_cube_state()

        for i in range(self.DATA_SIZE):
            action = self.agent.policy(state)

            if epoch == 0:
                action = random.randint(0, 6), "clockwise" if random.random() < 0.5 else "counterclockwise"

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
