import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb

from keras.models import load_model

class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()
        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.buffer.load()

        self.STEPS = 5
        self.DATA_SIZE = len(self.buffer.df)

    def train_model(self):
        for step in range(self.STEPS):
            batch = self.buffer.sample_gameplay_batch(int(self.DATA_SIZE * 0.1))
            self.agent.train(batch)

        self.agent.model.save("model.h5")

    def get_train_data(self):
        self.cube.scramble()
        state = self.cube.get_cube_state()
        done = False

        while not done:
            action = self.agent.policy(state, model, train=False)

            next_state, reward, done = self.cube.step(action)

            print("Action: ", action)
            print("Reward: ", self.cube.get_reward_state(state))

            # self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = next_state

            if done:
                print("Solved!")
                break

        self.buffer.save()


if __name__ == "__main__":
    model = load_model("model.h5")
    m = Main()
    m.get_train_data()