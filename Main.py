from Board import Board
from Robby import Robby
import math
import numpy as np
import random


def get_starting_position(rows, cols):
    row = random.randint(0, rows - 1)
    col = random.randint(0, cols - 1)
    return row, col
    return 0, 0


def do_episode(robby, actions, epsilon):
    robby.epsilon = epsilon
    for action_index in range(actions):
        action = robby.choose_action()
        robby.do_action(action)
        robby.display()
    return robby.total_reward, robby.wall_hits, robby.good_pick_ups, robby.bad_pick_ups


def do_experiment(name, row_count, col_count, episode_count, actions_per_episode, epsilon, change_epsilon,
                  learning_rate, discount_rate, can_probability, action_tax, testing, robby):
    # initalize robby
    if not testing:
        robby = Robby(learning_rate, discount_rate, show_move=False, action_tax=action_tax)
    training_rewards = []
    training_wall_hits = []
    training_good_pick_ups = []
    training_bad_pickups = []
    total_rewards = 0
    total_varience = 0
    # train robby
    for episode_index in range(episode_count):
        if change_epsilon:
            # decrement epsilon
            if episode_index % 50 == 0 and epsilon > 0.1:
                epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(robby, actions_per_episode, epsilon)
        total_rewards += episode_value
        total_varience += episode_value ** 2
        if episode_index % 100 == 0:
            training_rewards.append(episode_value)
            training_wall_hits.append(wall_hits)
            training_good_pick_ups.append(good_pick_ups)
            training_bad_pickups.append(bad_pickups)
    print name + " rewards " + str(training_rewards)
    print "wall hits: " + str(training_wall_hits)
    print "good pickups: " + str(training_good_pick_ups)
    print "bad pickups: " + str(training_bad_pickups)
    print "total wall hits: " + str(sum(training_wall_hits))
    print "total good pickups: " + str(sum(training_good_pick_ups))
    print "total bad pickups: " + str(sum(training_bad_pickups))
    print name + " rewards sum " + str(total_rewards)
    print name + " rewards average " + str(total_rewards/episode_count)
    print name + " rewards standard deviation " + str(math.sqrt(total_varience/episode_count))
    print
    print
    np.save(name, np.array(training_rewards))
    return robby

if __name__ == "__main__":
    # epsilon starts at 1.1 because its decremented on the first pass
    robby = do_experiment("train1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("test1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("learning_rate_train.25", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.25, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("learning_rate_test.25", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.25, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("learning_rate_train.50", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.5, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("learning_rate_test.50", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.5, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("learning_rate_train.75", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.75, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("learning_rate_test.75", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.75, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("learning_rate_train1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=1, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("learning_rate_test1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=1, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("constant_epsilon_train", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=.5, change_epsilon=False, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("constant_epsilon_test", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("action_tax_train", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=.5, change_epsilon=False, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=True, testing=False, robby=None)
    robby = do_experiment("action_tax_test", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=.9, can_probability=.5,
                          action_tax=True, testing=True, robby=robby)
    robby = do_experiment("discount_rate_train.25", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.2, discount_rate=.25, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("discount_rate_test.25", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=.25, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("discount_rate_train.50", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.5, discount_rate=.5, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("discount_rate_test.50", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.5, discount_rate=.5, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("discount_rate_train.75", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.2, discount_rate=.75, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("discount_rate_test.75", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=.75, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
    robby = do_experiment("discount_rate_train1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=1.1, change_epsilon=True, learning_rate=.2, discount_rate=1, can_probability=.5,
                          action_tax=False, testing=False, robby=None)
    robby = do_experiment("discount_rate_test1", row_count=10, col_count=10, episode_count=5000, actions_per_episode=200,
                          epsilon=0, change_epsilon=False, learning_rate=.2, discount_rate=1, can_probability=.5,
                          action_tax=False, testing=True, robby=robby)
