from cube.helper_cube import CubeHelper
from dqn_agent import DqnAgent

from keras.models import load_model
import copy

model = load_model('../models/model2.h5')


def try_solve():
    cube = CubeHelper()
    agent = DqnAgent()

    cube.scramble()

    state = copy.deepcopy(cube.get_cube_state())

    for i in range(5000):
        action = agent.policy(agent.one_hot_encode(state), model)

        next_state, reward, done = cube.step(state, action)

        state = copy.deepcopy(next_state)

        print("Action: ", action)
        print("Reward: ", reward)
        print("Progress: ", cube.get_reward_state(next_state))

        if done:
            print("Solved!")
            break


try_solve()
