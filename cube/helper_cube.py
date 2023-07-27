import copy

from cube.rubiks_cube import RubiksCube

import random
import numpy as np


class CubeHelper(RubiksCube):
    def __init__(self):
        super().__init__()

        self.non_effecting_rotations = {
            "F": ["F", "B"],
            "B": ["B", "F"],
            "U": ["U", "D"],
            "D": ["D", "U"],
            "R": ["R", "L"],
            "L": ["L", "R"]
        }

    def scramble(self, scramble_length=100):
        for _ in range(scramble_length):
            self.rotate(random.choice(self.cube_rotations))

    def get_cube_state(self):
        return list(self.cube.values())

    def check_solved(self):
        state = np.array(self.get_cube_state())
        return np.all(state == state[:, 0, 0][:, None, None], axis=(1, 2)).all()

    def reset(self):
        self.__init__()
        return self.get_cube_state()

    def excluded_face_scramble(self):
        """
        Scrambles all nodes on the cube except for one face that will stay the same.
        :return: the face that was scrambled without one face except for 1 move on that face,
                also the action that needs to be taken to solve that face
        """
        fixed_face = random.choice(self.faces)

        for i in range(10):
            rotation = random.choice(self.non_effecting_rotations[fixed_face])
            self.rotate(rotation)

        effected_faces = [x for x in self.faces if x not in self.non_effecting_rotations[fixed_face]]
        counter = random.choice(["", "'"])
        random_effecting_action = random.choice(effected_faces) + counter
        solving_action = random_effecting_action + "'" if counter == "" else random_effecting_action

        self.rotate(random_effecting_action)
        state = copy.deepcopy(self.get_cube_state())

        return state, solving_action

    @staticmethod
    def reward_face_solved(state, next_state):
        """
        Calculate the reward for the current action based on solved faces
        :return reward between -6 and 6
        """
        state, next_state = np.array(state), np.array(next_state)

        # Calculate the number of solved faces
        solved_faces = np.sum(np.all(state == state[:, 0, 0][:, None, None], axis=(1, 2)))
        next_solved_faces = np.sum(np.all(next_state == next_state[:, 0, 0][:, None, None], axis=(1, 2)))

        return next_solved_faces - solved_faces

    def reward_color_count(self, state):
        """
        Calculate the reward for the current action based on number of corresponding colors on each face
        :return reward between 0 and 1
        """
        reward = 0
        for ind, face in enumerate(state):
            face = [x for y in face for x in y]  # Flatten list
            face_color = self.colors[ind]
            reward += face.count(face_color)

        return reward / 54

    def step(self, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        state = copy.deepcopy(self.get_cube_state())
        self.rotate(action)

        next_state = copy.deepcopy(self.get_cube_state())
        reward = self.reward_action(state, next_state)
        done = self.check_solved()

        return next_state, reward, done

    def reward_action(self, state, next_state):
        reward = 10 * (self.reward_color_count(next_state) - self.reward_color_count(state))

        solved_face = self.reward_face_solved(state, next_state)
        reward += solved_face

        if self.check_solved():
            print('Solved cube!')
            reward += 3

        move_penalty = -0.05

        return reward + move_penalty


if __name__ == "__main__":
    cube = CubeHelper()

