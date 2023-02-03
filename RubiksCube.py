class RubiksCube:
    def __init__(self):
        self.cube = {}
        self.faces = ["front", "right", "up", "bottom", "left", "down"]
        self.colors = ["G", "R", "W", "B", "O", "Y"]

        for i in range(6):
            self.cube[self.faces[i]] = [[self.colors[i] for _ in range(3)] for _ in range(3)]

    def __str__(self):
        return str(self.cube)


    def rotate(self, face, direction):
        if face == "front":
            self.rotate_face(["right", "up", "left", "down"], direction, 0)

    def rotate_face(self, affected_cubes, direction, position):
        if direction == "counterclockwise":
            affected_cubes = affected_cubes[::-1]  # Reverse list

        first_cube = self.cube[affected_cubes[0]][position]  # Store first cube to be used later

        for ind, cube_name in enumerate(affected_cubes):
            if ind == 3:  # If last cube
                print(self.cube[affected_cubes[0]])
                self.cube[cube_name][position] = first_cube
            else:
                self.cube[cube_name][position] = self.cube[affected_cubes[ind + 1]][position]


if __name__ == "__main__":
    r = RubiksCube()
    print(r, "\n\n")
    r.rotate("front", "clockwise")
    print(r)
