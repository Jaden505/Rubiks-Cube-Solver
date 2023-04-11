from keras.models import load_model

from cube import RubiksCube as rc
import DqnAgent

import copy

model = load_model('../models/model.h5')

def try_solve():
    cube = rc.RubiksCube()
    cube.scramble()

    state = copy.deepcopy(cube.get_cube_state())

    for i in range(5000):
        action = list(da.policy(da.one_hot_encode(state), model))
        action[1] = "clockwise" if action[1] == 0 else "counterclockwise"

        next_state, reward, done = cube.step(state, action)

        state = copy.deepcopy(next_state)

        print("Action: ", action)
        print("Reward: ", reward)
        print("Progress: ", cube.get_reward_state(next_state))

        if done:
            print("Solved!")
            break


da = DqnAgent.DqnAgent()
try_solve()
