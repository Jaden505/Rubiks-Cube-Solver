import copy

class RubiksCube:
    def __init__(self):
        self.cube = {}
        self.faces = ["U", "D", "F", "B", "R", "L"]
        self.directions = ["clockwise", "counterclockwise"]
        self.colors = [0, 1, 2, 3, 4, 5]
        self.cube_rotations = [
            "U", "U'",  # Upper face rotations
            "D", "D'",  # Down face rotations
            "F", "F'",  # Front face rotations
            "B", "B'",  # Back face rotations
            "R", "R'",  # Right face rotations
            "L", "L'"  # Left face rotations
        ]

        for i in range(6):
            self.cube[self.faces[i]] = [[self.colors[i] for _ in range(3)] for _ in range(3)]

        self.solved_state = copy.deepcopy(self.cube)

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

    def rotate(self, rotation_str):
        """
            Rotate a row or column of the cube depending on the input string
            :param rotation_str: The string to rotate the cube based on one of the strings in `cube_rotations`
        """

        if rotation_str not in self.cube_rotations:
            raise Exception("Invalid rotation string given")

        face = rotation_str[0]
        direction = "clockwise"
        if len(rotation_str) > 1:
            if rotation_str[1] == "2":
                self.rotate(rotation_str[0] + "'")
                self.rotate(rotation_str[0] + "'")
                return self.cube
            elif rotation_str[1] == "'":
                direction = "counterclockwise"

        if face == "U":
            self.rotate_face(["R", "F", "L", "B"], direction, 0, "rows")
        elif face == "D":
            self.rotate_face(["R", "B", "L", "F"], direction, 2, "rows")
        elif face == "F":
            self.rotate_face(["U", "R", "D", "L"], direction, 0, "columns")
        elif face == "B":
            self.rotate_face(["U", "L", "D", "R"], direction, 2, "columns")
        elif face == "R":
            self.rotate_face(["U", "B", "D", "F"], direction, 2, "columns")
        elif face == "L":
            self.rotate_face(["U", "F", "D", "B"], direction, 0, "columns")
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

    def __str__(self):
        return str(self.cube)


if __name__ == "__main__":
    rb = RubiksCube()
    print(rb.cube)
