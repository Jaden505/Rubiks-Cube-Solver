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
            self.rotate_face(["right", "up", "left", "down"], direction, 0, True)
        elif face == "right":
            self.rotate_face(["front", "down", "bottom", "up"], direction, 2, False)
        elif face == "up":
            self.rotate_face(["front", "right", "bottom", "left"], direction, 0, False)
        elif face == "bottom":
            self.rotate_face(["up", "right", "down", "left"], direction, 2, True)
        elif face == "left":
            self.rotate_face(["front", "up", "bottom", "down"], direction, 0, False)
        elif face == "down":
            self.rotate_face(["front", "left", "bottom", "right"], direction, 2, False)

    def rotate_face(self, affected_cubes, direction, position, rows):
        if direction == "counterclockwise":
            affected_cubes = affected_cubes[::-1]  # Reverse list

        if rows:
            first_cube = self.cube[affected_cubes[0]][position]  # Store first cube to be used later

            for ind, cube_name in enumerate(affected_cubes):
                if ind == 3:  # If last cube
                    self.cube[cube_name][position] = first_cube
                else:
                    self.cube[cube_name][position] = self.cube[affected_cubes[ind + 1]][position]

        else:
            first_cube = [x[position] for x in self.cube[affected_cubes[0]]]  # Store first cube to be used later

            for ind, cube_name in enumerate(affected_cubes):
                if ind == 3:  # If last cube
                    for i in range(3):
                        self.cube[cube_name][i][position] = first_cube[i]
                else:
                    next_cube = [x[position] for x in self.cube[affected_cubes[ind + 1]]]
                    for i in range(3):
                        self.cube[cube_name][i][position] = next_cube[i]

if __name__ == "__main__":
    r = RubiksCube()
    print(r, "\n\n")
    print(r)989
