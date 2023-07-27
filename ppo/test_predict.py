from cube.helper_cube import CubeHelper
from agent import PPOAgent
from main import flatten_state

from keras.models import load_model
import copy
import numpy as np

cube = CubeHelper()
agent = PPOAgent(state_dim=54, action_dim=12)
agent.actor = load_model("../models/ppo/model.h5_actor")
agent.critic = load_model("../models/ppo/model.h5_critic")

rotation_dict = {0: "U", 1: "U'", 2: "D", 3: "D'", 4: "L", 5: "L'",
                      6: "R", 7: "R'", 8: "F", 9: "F'", 10: "B", 11: "B'"}

def try_solve():
    scramble_length = 20
    cube.scramble(scramble_length)
    state = flatten_state(cube.get_cube_state())

    for i in range(1000):
        action = agent.get_action(np.expand_dims(state, axis=0))
        next_state, reward, done = cube.step(rotation_dict[action])

        state = flatten_state(next_state)

        print("Action: ", action)
        print("Reward: ", reward)
        print("Progress: ", cube.reward_color_count(next_state))

        if done:
            print("Solved!")
            print("Steps: ", i)
            break

        if i % scramble_length == 0:
            cube.reset()
            cube.scramble(scramble_length)
            state = flatten_state(cube.get_cube_state())


try_solve()
