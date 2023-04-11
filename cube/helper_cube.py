from rubiks_cube import RubiksCube

import random


class CubeHelper(RubiksCube):
    def scramble(self):
        for _ in range(100):
            face = random.choice(self.faces)
            direction = random.choice(self.directions)
            self.rotate(face, direction)

    def get_cube_state(self):
        return list(self.cube.values())

    def get_reward_state(self, state):
        reward = 0
        for face in state:
            face = [x for y in face for x in y]  # Flatten list
            reward += max([face.count(x) for x in self.colors])  # Get max count of each color

        return reward

    def check_solved(self):
        return self.get_reward_state(self.get_cube_state()) == 54

    def step(self, state, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        action_face = self.faces[action[0]]  # Get face name from action
        action_direction = action[1]  # Get direction name from action

        self.rotate(action_face, action_direction)

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

    def __str__(self):
        return str(self.cube)
