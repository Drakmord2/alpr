import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import cv2 as cv


class Representation:
    def __init__(self, image, image_name):
        self.image = image
        self.image_name = image_name
        self.borders = []

    def chain_code(self):
        start_point = self.get_start_point()

        if start_point == ():
            return False

        change_j = [-1, 0, 1,
                    -1,    1,
                    -1, 0, 1]

        change_i = [-1, -1, -1,
                     0,      0,
                     1,  1,  1]

        directions = [0, 1, 2,
                      7,    3,
                      6, 5, 4]

        dir2idx = dict(zip(directions, range(len(directions))))

        border = []
        chain = []
        curr_point = start_point
        for direction in directions:
            idx = dir2idx[direction]
            new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])

            if self.image[new_point] != 0:
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break

        count = 0
        while curr_point != start_point:
            b_direction = (direction + 5) % 8
            dirs_1 = range(b_direction, 8)
            dirs_2 = range(0, b_direction)
            dirs = []
            dirs.extend(dirs_1)
            dirs.extend(dirs_2)

            for direction in dirs:
                idx = dir2idx[direction]
                new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])

                if self.image[new_point] != 0:
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break

            if count == 1000:
                break

            count += 1

        self.borders.append(border)

        plt.imshow(self.image, cmap='Greys')
        plt.plot([i[1] for i in border], [i[0] for i in border])
        plt.savefig('../bin/' + self.image_name + '-chain.png')

        return True

    def get_start_point(self):
        xsize, ysize = self.image.shape
        borders_x = set()
        borders_y = set()

        for borders in self.borders:
            borders_x.update({v[0] for v in borders})
            borders_y.update({v[1] for v in borders})

        start_point = ()
        for i, row in enumerate(self.image):
            for j, value in enumerate(row):
                if i in borders_x and j in borders_y:
                    continue

                if i+1 in borders_x and j+1 in borders_y:
                    continue

                if i-1 in borders_x and j-1 in borders_y:
                    continue

                if i < 1 or i > xsize - 2:
                    continue

                if j < 1 or i > ysize - 2:
                    continue

                if value == 255:
                    start_point = (i, j)
                    break
            else:
                continue

            break

        return start_point
