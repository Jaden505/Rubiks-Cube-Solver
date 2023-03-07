import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb

class Main:
    def __init__(self):
        self.cube = rc.RubiksCube()
        self.agent = da.DqnAgent()
        self.buffer = rb.ReplayBuffer()

        self.buffer.load()

        self.STEPS = 500
        self.DATA_SIZE = len(self.buffer.df)

    def train_model(self):
        for step in range(self.STEPS):
            batch = self.buffer.sample_gameplay_batch(int(self.DATA_SIZE * 0.8))
            self.agent.train(batch)

        self.agent.model.save("model.h5")

    def get_train_data(self):
        self.cube.scramble()
        state = self.cube.get_cube_state()

        for i in range(self.DATA_SIZE):
            action = self.agent.policy(state, self.agent.target_model, train=False)

            next_state, reward, done = self.cube.step(action)

            print("Action: ", action)
            print("Reward: ", reward)

            self.buffer.add_gameplay(state, next_state, reward, action, done)
            state = next_state

            if done:
                print("Solved!")
                break

        self.buffer.save()


if __name__ == "__main__":
    m = Main()
    m.train_model()