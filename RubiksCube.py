import random

class RubiksCube:
    def __init__(self):
        self.cube = {}
        self.faces = ["front", "right", "up", "bottom", "left", "down"]
        self.colors = ["G", "R", "W", "B", "O", "Y"]

        for i in range(6):
            self.cube[self.faces[i]] = [[self.colors[i] for _ in range(3)] for _ in range(3)]

    def __str__(self):
        return str(self.cube)

    def rotateRows(self, affected_cubes, position):
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

    def rotateColumns(self, affected_cubes, position):
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
            self.rotateFace(["right", "up", "left", "down"], direction, 0, "rows")
        elif face == "right":
            self.rotateFace(["front", "down", "bottom", "up"], direction, 2, "columns")
        elif face == "up":
            self.rotateFace(["front", "right", "bottom", "left"], direction, 0, "columns")
        elif face == "bottom":
            self.rotateFace(["up", "right", "down", "left"], direction, 2, "rows")
        elif face == "left":
            self.rotateFace(["front", "up", "bottom", "down"], direction, 0, "columns")
        elif face == "down":
            self.rotateFace(["front", "left", "bottom", "right"], direction, 2, "columns")
        else:
            raise Exception("Invalid face name given")

        return self.cube

    def rotateFace(self, affected_cubes, direction, position, rotation):
        if direction == "counterclockwise":
            affected_cubes = affected_cubes[::-1]  # Reverse list

        if rotation == "rows":
            self.rotateRows(affected_cubes, position)
        else:
            self.rotateColumns(affected_cubes, position)

    def scramble(self):
        directions = ["clockwise", "counterclockwise"]

        for _ in range(100):
            face = random.choice(self.faces)
            direction = random.choice(directions)
            self.rotate(face, direction)

if __name__ == "__main__":
    r = RubiksCube()
    print(r)
    r.scramble()
    print(r)
