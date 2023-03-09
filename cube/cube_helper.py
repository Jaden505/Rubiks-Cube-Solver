import RubiksCube as rc

import copy


class CubeHelper(rc.RubiksCube):
    def __init__(self):
        super().__init__()

    def get_cube_state(self):
        return list(super().cube.values())

    def get_reward_state(self, state):
        reward = 0
        for face in state:
            face = [x for y in face for x in y]  # Flatten list
            reward += max([face.count(x) for x in super().cube.colors])  # Get max count of each color

        return reward / 54  # normalize reward


    def check_solved(self):
        return self.get_reward_state(self.get_cube_state()) == 54


    def step(self, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        first_state = copy.deepcopy(self.get_cube_state())  # Store first state to calculate reward later
        action_face = super().cube.faces[action[0]]  # Get face name from action
        action_direction = action[1]  # Get direction name from action

        self.cube.rotate(action_face, action_direction)

        next_state = self.get_cube_state()
        reward = self.get_reward_action(first_state, next_state)
        done = self.check_solved()

        return next_state, reward, done


    def get_reward_action(self, state, next_state):
        max_action_reward = 12 / 54
        improvement = self.get_reward_state(next_state) - self.get_reward_state(state)
        improvement = 0 if improvement < 0 else improvement  # Set negatives to 0

        return improvement / max_action_reward
