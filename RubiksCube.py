import random


class RubiksCube:
    def __init__(self):
        self.cube = {}
        self.faces = ["front", "right", "up", "bottom", "left", "down"]
        self.colors = [0,1,2,3,4,5,6]
        self.directions = ["clockwise", "counterclockwise"]

        for i in range(6):
            self.cube[self.faces[i]] = [[self.colors[i] for _ in range(3)] for _ in range(3)]

    def __str__(self):
        return str(self.cube)

    def rotate_rows(self, affected_cubes, position):
        """
        Rotate the rows of the cubes given
        :param affected_cubes: The cubes to rotate
        :param position: The position of the rows to rotate
        """

        # Store first cube by getting the row of the first cube at the given position
        first_cube = self.cube[affected_cubes[0]][position]

        for ind, cube_name in enumerate(affected_cubes):
            if ind == 3:  # If last cube
                self.cube[cube_name][position] = first_cube
            else:
                self.cube[cube_name][position] = self.cube[affected_cubes[ind + 1]][position]

    def rotate_columns(self, affected_cubes, position):
        """
        Rotate the columns of the cubes given
        :param affected_cubes: The cubes to rotate
        :param position: The position of the columns to rotate
        """

        # Store first cube by getting the column of the first cube at the given position
        first_cube = [x[position] for x in self.cube[affected_cubes[0]]]

        for ind, cube_name in enumerate(affected_cubes):
            if ind == 3:  # If last cube
                for i in range(3):
                    self.cube[cube_name][i][position] = first_cube[i]
            else:
                next_cube = [x[position] for x in self.cube[affected_cubes[ind + 1]]]
                for i in range(3):
                    self.cube[cube_name][i][position] = next_cube[i]

    def rotate(self, face, direction):
        """
            Rotate a row or column of the cube depending on the face given
            :param face: The face of the cube to rotate (front, right, up, bottom, left, down)
            :param direction: The direction to rotate the face (clockwise or counterclockwise)
        """

        if face == "front":
            self.rotate_face(["right", "up", "left", "down"], direction, 0, "rows")
        elif face == "right":
            self.rotate_face(["front", "down", "bottom", "up"], direction, 2, "columns")
        elif face == "up":
            self.rotate_face(["front", "right", "bottom", "left"], direction, 0, "columns")
        elif face == "bottom":
            self.rotate_face(["up", "right", "down", "left"], direction, 2, "rows")
        elif face == "left":
            self.rotate_face(["front", "up", "bottom", "down"], direction, 0, "columns")
        elif face == "down":
            self.rotate_face(["front", "left", "bottom", "right"], direction, 2, "columns")
        else:
            raise Exception("Invalid face name given")

        return self.cube

    def rotate_face(self, affected_cubes, direction, position, rotation):
        if direction == "counterclockwise":
            affected_cubes = affected_cubes[::-1]  # Reverse list

        if rotation == "rows":
            self.rotate_rows(affected_cubes, position)
        else:
            self.rotate_columns(affected_cubes, position)

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

    def step(self, action):
        """
        Rotate the cube and return the next state, reward and if the cube is solved
        """
        first_state = self.get_cube_state()  # Store first state to calculate reward later
        action_face = self.faces[action[0]]  # Get face name from action
        action_direction = self.directions[action[1]]  # Get direction name from action

        self.rotate(action_face, action_direction)

        next_state = self.get_cube_state()
        reward = self.get_reward_action(first_state, next_state)
        done = self.check_solved()

        return next_state, reward, done

    def get_reward_action(self, state, next_state):
        return self.get_reward_state(next_state) - self.get_reward_state(state)

if __name__ == "__main__":
    rb = RubiksCube()
    rb.scramble()
    print(rb)
