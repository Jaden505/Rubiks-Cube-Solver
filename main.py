import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb


def train_model():
    cube = rc.RubiksCube()
    agent = da.DqnAgent()
    buffer = rb.ReplayBuffer()

    for i in range(2000):
        get_train_data(cube, agent, buffer)
        batch = buffer.sample_gameplay_batch(32)
        loss = agent.train(batch)
        print("Loss: ", loss)

        # if i % 20 == 0:  # Update neural network every 20 iterations
        #     agent.update_network()


def get_train_data(cube, agent, buffer):
    cube.scramble()
    state = cube.get_cube_state()

    for i in range(100):
        action = agent.policy(state)
        next_state, reward, done = cube.step(action)
        buffer.add_gameplay(state, next_state, reward, action, done)
        state = next_state
        if done:
            break


if __name__ == "__main__":
    train_model()