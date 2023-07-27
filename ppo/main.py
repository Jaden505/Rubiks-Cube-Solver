from cube.helper_cube import CubeHelper
from agent import PPOAgent

import numpy as np
from keras.models import load_model


class Main:
    def __init__(self):
        self.cube = CubeHelper()
        self.agent = PPOAgent(state_dim=54, action_dim=12)

        self.STEPS = 100000
        self.BATCH_SIZE = 64

        self.model_save_path = "../models/ppo/model.h5"
        self.agent.actor = load_model(self.model_save_path + "_actor")
        self.agent.critic = load_model(self.model_save_path + "_critic")
        self.agent.epsilon = 0.5

        self.solved_count = 0
        self.scramble_length = 7

        self.rotation_dict = {0: "U", 1: "U'", 2: "D", 3: "D'", 4: "L", 5: "L'",
                              6: "R", 7: "R'", 8: "F", 9: "F'", 10: "B", 11: "B'"}

    def train_model(self):
        for step in range(self.STEPS):
            states, actions, rewards, next_states, dones = self.get_train_data()

            self.agent.train(np.array(states), np.array(actions), np.array(rewards), np.array(next_states),
                             np.array(dones))

            if step % 10 == 0:
                self.agent.actor.save(self.model_save_path + "_actor")
                self.agent.critic.save(self.model_save_path + "_critic")

            print("Total reward: ", sum(rewards))

            if self.scramble_length >= 25:
                self.scramble_length = 2

        self.agent.actor.save(self.model_save_path + "_actor")
        self.agent.critic.save(self.model_save_path + "_critic")

    def get_train_data(self):
        self.cube.reset()
        self.cube.scramble(self.scramble_length)
        state = flatten_state(self.cube.get_cube_state())
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in range(self.BATCH_SIZE):
            action = self.agent.get_action(np.expand_dims(state, axis=0))
            string_rotation = self.rotation_dict[action]
            next_state, reward, done = self.cube.step(string_rotation)
            next_state = flatten_state(next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = flatten_state(next_state)

            if done:
                if self.solved_count >= 10:
                    self.scramble_length += 1
                    self.solved_count = 0
                    self.agent.epsilon = 1.0
                else:
                    self.solved_count += 1

            if i % self.scramble_length == 0 or done:
                self.cube.reset()
                self.cube.scramble(self.scramble_length)
                state = flatten_state(self.cube.get_cube_state())

        return states, actions, rewards, next_states, dones


def flatten_state(state):
    """
    Input is a 6x3x3 array of the Rubik's cube
    Output is a flattened array of the Rubik's cube of shape (54,)
    """
    flat_state = np.array(state).flatten()
    return flat_state.astype(int)


if __name__ == "__main__":
    m = Main()
    m.train_model()
