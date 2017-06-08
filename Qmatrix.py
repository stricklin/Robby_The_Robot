import numpy as np


class Qmatrix:

    def __init__(self, sensor_count, sensor_value_count, action_count, learning_rate, discount_rate):
        """
        initalizes a Qmatrix with all values set to 0.
        the states are indexed: north^1 + east^2 + south^3 + west^4 + here^5
        the actions are indexed:
            pick up can = 0
            move north = 1
            move east = 2
            move south = 3
            move west = 4
        :param learning_rate: the learning rate
        :param discount_rate: the discount rate
        """
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.sensor_count = sensor_count
        self.sensor_value_count = sensor_value_count
        self.action_count = action_count
        self.state_count = self.get_state_count()
        self.matrix = np.zeros((self.state_count, self.action_count))

    def get_state_count(self):
        state_count = 0
        for sensor_index in range(1, self.sensor_count + 1):
            state_count += self.sensor_value_count ** sensor_index
        return state_count

    @ staticmethod
    def get_state_index(state):
        return int(state[0] ** 1 + state[1] ** 2 + state[2] ** 3 + state[3] ** 4 + state[4] ** 5) - 1

    def get_value(self, state, action_index):
        state_index = self.get_state_index(state)
        return self.matrix[state_index][action_index]

    def get_highest_state_value(self, state):
        highest_value = None
        for action in range(self.action_count):
            value = self.get_value(state, action)
            if highest_value is None or highest_value < value:
                highest_value = value
        return highest_value

    def update(self, old_state, action_index, reward, new_state):
        state_index = self.get_state_index(old_state)
        old_value = self.matrix[state_index][action_index]
        new_value = old_value + self.learning_rate *\
                                (reward + self.discount_rate * self.get_highest_state_value(new_state)) - old_value
        self.matrix[state_index][action_index] = new_value
