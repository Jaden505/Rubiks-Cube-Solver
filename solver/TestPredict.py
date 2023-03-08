from keras.models import load_model
from numpy import array

from cube import RubiksCube as rc
import DqnAgent


model = load_model('../models/model.h5')

def try_solve():
    cube = rc.RubiksCube()
    cube.scramble()

    state = cube.get_cube_state()
    done = False


    while not done:
        action = da.policy(state, model)
        next_state, reward, done = cube.step(action)

        print("Reward: ", reward)

        state = next_state

        if done:
            print("Solved!")
            break


da = DqnAgent.DqnAgent()
try_solve()
