from cube.rubiks_cube import RubiksCube

import random
import numpy as np


class CubeHelper(RubiksCube):
    def __init__(self):
        super().__init__()

    def scramble(self):
        for _ in range(100):
            self.rotate(random.choice(self.cube_rotations))

    def get_cube_state(self):
        return list(self.cube.values())

    @staticmethod
    def reward_function(state, next_state):
        """
        Calculate the reward for the current action based on solved faces
        :return reward between -6 and 6
        """
        state, next_state = np.array(state), np.array(next_state)

        # Calculate the number of solved faces
        solved_faces = np.sum(np.all(state == state[:, 0, 0][:, None, None], axis=(1, 2)))
        next_solved_faces = np.sum(np.all(next_state == next_state[:, 0, 0][:, None, None], axis=(1, 2)))
        print(solved_faces, next_solved_faces)

        return next_solved_faces - solved_faces

    def check_solved(self):
        state = np.array(self.get_cube_state())
        return np.all(state == state[:, 0, 0][:, None, None], axis=(1, 2)).all()

    def step(self, state, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        self.rotate(action)

        next_state = self.get_cube_state()
        reward = CubeHelper.reward_function(state, next_state)
        done = self.check_solved()

        return next_state, reward, done

    def reset(self):
        self.__init__()
        return self.get_cube_state()


if __name__ == "__main__":
    cube = CubeHelper()
