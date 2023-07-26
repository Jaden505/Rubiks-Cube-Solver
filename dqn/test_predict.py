from cube.helper_cube import CubeHelper
from dqn_agent import DqnAgent

from keras.models import load_model
import copy

cube = CubeHelper()
agent = DqnAgent()
model = load_model('../models/dqn/model.h5')

agent.model = model


def try_solve():
    scramble_length = 6
    cube.scramble(scramble_length)
    state = copy.deepcopy(cube.get_cube_state())

    for i in range(1000):
        ohe_state = agent.one_hot_encode(state)
        action, q_state = agent.boltzmann_exploration(ohe_state, agent.model)
        next_state, reward, done = cube.step(agent.rotation_dict[action], i, scramble_length)

        state = copy.deepcopy(next_state)

        print("Action: ", action)
        print("Reward: ", reward)
        print("Progress: ", cube.reward_color_count(state) * 54)

        if done:
            print("Solved!")
            print("Steps: ", i)
            break

        if i % scramble_length == 0:
            cube.reset()
            cube.scramble(scramble_length)
            state = copy.deepcopy(cube.get_cube_state())


try_solve()
