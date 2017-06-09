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
    return robby.total_reward, robby.wall_hits, robby.good_pick_ups, robby.bad_pick_ups


def do_experiment(name, row_count, col_count, episode_count, actions_per_episode, epsilon, change_epsilon,
                  learning_rate, discount_rate, can_probability, action_tax, testing, robby):
    # initalize robby
    if not testing:
        robby = Robby(learning_rate, discount_rate, show_move=False, action_tax=action_tax)
    reward_plots = []
    rewards = []
    wall_hits = []
    good_pickups = []
    bad_pickups = []
    # run episodes
    for episode_index in range(episode_count):
        if change_epsilon:
            # decrement epsilon
            if episode_index % 50 == 0 and epsilon > 0.1:
                epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hit, good_pickup, bad_pickup = do_episode(robby, actions_per_episode, epsilon)
        rewards.append(episode_value)
        wall_hits.append(wall_hit)
        good_pickups.append(good_pickup)
        bad_pickups.append(bad_pickup)
        if episode_index % 100 == 0:
            reward_plots.append(episode_count)
    rewards = np.array(rewards)
    wall_hits = np.array(wall_hits)
    good_pickups = np.array(good_pickups)
    bad_pickups = np.array(bad_pickups)
    print name + " rewards " + str(rewards)
    print "wall hits: " + str(wall_hits)
    print "good pickups: " + str(good_pickups)
    print "bad pickups: " + str(bad_pickups)
    print "total wall hits: " + str(sum(wall_hits))
    print "total good pickups: " + str(sum(good_pickups))
    print "total bad pickups: " + str(sum(bad_pickups))
    total_rewards = sum(rewards)
    print name + " rewards sum " + str(total_rewards)
    rewards_average = total_rewards/episode_count
    print name + " rewards average " + str(rewards_average)
    print name + " rewards standard deviation " + str(np.std(rewards))
    print
    print
    np.save(name, np.array(reward_plots))
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

