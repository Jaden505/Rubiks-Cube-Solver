from cube.helper_cube import CubeHelper
from dqn_agent import DqnAgent

from keras.models import load_model
import copy

cube = CubeHelper()
agent = DqnAgent()
model = load_model('../models/model.h5')

agent.model = model

def try_solve():
    cube.scramble()
    state = cube.get_cube_state()

    for i in range(5000):
        ohe_state = agent.one_hot_encode(state)
        action, q_state = agent.boltzmann_exploration(ohe_state, agent.model)
        next_state, reward, done = cube.step(agent.rotation_dict[action])

        print("Action: ", action)
        print("Reward: ", reward)
        print("Progress", cube.reward_color_count(state))
        print("State: ", state)

        state = copy.deepcopy(next_state)

        if done:
            print("Solved!")
            break

try_solve()
