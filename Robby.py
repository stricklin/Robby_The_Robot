import random
import os
import time
from Qmatrix import Qmatrix


class Robby:

    def __init__(self, learning_rate, discount_rate, show_move, action_tax):
        self.board = None
        self.row = None
        self.col = None
        self.epsilon = None
        self.show_move = show_move
        self.action_tax = action_tax
        self.sensor_count = 5
        self.sensor_value_count = 3
        self.action_count = 5
        self.qmatrix = Qmatrix(self.sensor_count, self.sensor_value_count, self.action_count,
                               learning_rate, discount_rate)
        self.training = True
        self.total_reward = 0
        self.wall_hits = 0
        self.good_pick_ups = 0
        self.bad_pick_ups = 0

    def new_episode(self, board, starting_position, epsilon):
        self.board = board
        self.row = starting_position[0]
        self.col = starting_position[1]
        self.epsilon = epsilon
        self.total_reward = 0
        self.wall_hits = 0
        self.good_pick_ups = 0
        self.bad_pick_ups = 0

    def sense(self):
        """
        robby senses his environment
        north, east, south, and west are all square adjacent to robby
        here is the square he currently occupies
        squares can have the values:
            empty sqaure = 0
            square containing can = 1
            wall = 2
        :return: row, col, and the values of north, east, south, west, and here
        """
        north = self.board.get_square(self.row - 1, self.col)
        east = self.board.get_square(self.row, self.col + 1)
        south = self.board.get_square(self.row + 1, self.col)
        west = self.board.get_square(self.row, self.col - 1)
        here = self.board.get_square(self.row, self.col)
        return north, east, south, west, here

    def choose_action(self):
        # get actions
        actions = self.get_action_values()
        if random.random() > 1 - self.epsilon:
            # if taking a random action
            random.shuffle(actions)
            return actions[0][0]
        # if finding the best action
        best_actions = [actions[0]]
        for action in actions:
            if action[1] > best_actions[0][1]:
                best_actions = [action]
            if action[1] == best_actions[0][1]:
                best_actions.append(action)
        random.shuffle(best_actions)
        return best_actions[0][0]

    def get_action_values(self):
        state = self.sense()
        action_values = []
        for action in range(self.action_count):
            action_values.append((action, self.qmatrix.get_value(state, action)))
        return action_values

    def do_action(self, action):
        """
        robby performs the action
        and updates the Q matrix
        :param action: the action code
        :return: None
        """
        reward = 0
        if self.action_tax:
            reward = -0.5
        old_state = self.sense()
        if action == 0:
            # pick up can
            if self.board.pick_up_can(self.row, self.col):
                # successful can pick up
                reward += 10
                self.good_pick_ups += 1
            else:
                # failed can pick up
                reward += -1
                self.bad_pick_ups += 1
        else:
            # move in direction
            if action == 1:
                # north
                dest_row = self.row - 1
                dest_col = self.col
            elif action == 2:
                # east
                dest_row = self.row
                dest_col = self.col + 1
            elif action == 3:
                # south
                dest_row = self.row + 1
                dest_col = self.col
            elif action == 4:
                # west
                dest_row = self.row
                dest_col = self.col - 1

            if self.board.get_square(dest_row, dest_col) == 2:
                # hit a wall
                reward += -5
                self.wall_hits += 1
            else:
                # move to new square
                self.row = dest_row
                self.col = dest_col
        new_state = self.sense()
        self.total_reward += reward
        if self.training:
            self.qmatrix.update(old_state, action, reward, new_state)

    def display(self):
        if not self.show_move:
            return
        os.system('cls' if os.name == 'nt' else'clear')
        for row in range(self.board.row_count):
            line = []
            for col in range(self.board.col_count):
                if row == self.row and col == self.col:
                    line.append("R")
                else:
                    line.append(self.board.get_square(row, col))
            print line
        print "total reward: " + str(self.total_reward)
        # show index and match with computed index
        state = self.sense()
        print state

        time.sleep(60)
