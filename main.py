import RubiksCube as rc
import DqnAgent as da
import ReplayBuffer as rb

def train_model():
    cube = rc.RubiksCube()
    agent = da.DqnAgent()
    buffer = rb.ReplayBuffer()

    for i in range(1000):
        cube.reset()
        cube.scramble()
        state = cube.get_cube_state()
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = cube.step(action)
            buffer.add_gameplay(state, next_state, reward, action, done)
            state = next_state

        batch = buffer.sample_gameplay_batch(32)
        loss = agent.train(batch)
        print("Loss: ", loss)

def get_train_data(cube, agent, buffer):

    return []