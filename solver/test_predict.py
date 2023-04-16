from cube.helper_cube import CubeHelper
from dqn_agent import DqnAgent

from keras.models import load_model
import copy

cube = CubeHelper()
agent = DqnAgent()
model = load_model('../models/model.h5')


def try_solve():
    cube.scramble()

    for i in range(5000):
        action = agent.policy(agent.one_hot_encode(cube.get_cube_state()), model)
        cube.rotate(action)

        print("Action: ", action)
        print("Progress: ", cube.reward_color_count(cube.get_cube_state()))

        # if done:
        #     print("Solved!")
        #     break


try_solve()
