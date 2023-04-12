from cube.rubiks_cube import RubiksCube

import random


class CubeHelper(RubiksCube):
    def __init__(self):
        super().__init__()

    def scramble(self):
        for _ in range(100):
            self.rotate(random.choice(self.cube_rotations))

    def get_cube_state(self):
        return list(self.cube.values())

    def get_reward_state(self, cube):
        reward = 0

        reward += sum(row.count(0) for row in cube['U'])
        reward += sum(row.count(1) for row in cube['D'])
        reward += sum(row.count(2) for row in cube['F'])
        reward += sum(row.count(3) for row in cube['B'])
        reward += sum(row.count(4) for row in cube['R'])
        reward += sum(row.count(5) for row in cube['L'])

        return reward

    def check_solved(self):
        return self.get_reward_state(self.get_cube_state()) == 54

    def step(self, state, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        self.rotate(action)

        next_state = self.get_cube_state()
        reward = self.get_reward_action(state, next_state)
        done = self.check_solved()

        return next_state, reward, done

    def get_reward_action(self, state, next_state):
        reward = self.get_reward_state(next_state) - self.get_reward_state(state)
        return (reward / 12) + 1  # reward between 0 and 2

    def reset(self):
        self.__init__()
        return self.get_cube_state()

