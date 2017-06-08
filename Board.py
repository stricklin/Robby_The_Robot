import numpy as np
import random


class Board:
    """
    a board for robby to walk around on
    """
    def __init__(self, rows, columns, can_probability):
        self.row_count = rows
        self.col_count = columns
        self.can_prob = can_probability
        self.board = np.ones((self.row_count, self.col_count))
        self.place_cans()

    def place_cans(self):
        for row in range(self.row_count):
            for col in range(self.col_count):
                if random.random() < self.can_prob:
                    self.board[row][col] = 2

    def get_square(self, row, col):
        """
        get the value of a square
        empty sqaure = 0
        square containing can = 1
        wall = 2
        :param row: the row of the square
        :param col: the column of the square
        :return: the value of the square
        """
        # check for walls
        if row < 0 or row >= self.row_count:
            return 3
        if col < 0 or col >= self.col_count:
            return 3
        return self.board[row][col]

    def pick_up_can(self, row, col):
        if self.board[row][col] == 1:
            self.board[row][col] = 2
            return True
        return False


