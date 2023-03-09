from keras.models import load_model

from cube import RubiksCube as rc
import DqnAgent


model = load_model('../models/model.h5')

def try_solve():
    cube = rc.RubiksCube()
    cube.scramble()

    state = cube.get_cube_state()
    done = False


    while not done:
        face_index, direction = da.policy(state, model)
        next_state, reward, done = cube.step((face_index, direction))

        print("Action: ", (face_index, direction))
        print("Reward: ", reward)
        print("Progress: ", cube.get_reward_state(next_state))

        state = next_state

        if done:
            print("Solved!")
            break


da = DqnAgent.DqnAgent()
try_solve()
