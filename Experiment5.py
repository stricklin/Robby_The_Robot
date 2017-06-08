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


def do_episode(actions, epsilon):
    robby.epsilon = epsilon
    for action_index in range(actions):
        action = robby.choose_action()
        robby.do_action(action)
        robby.display()
    return robby.total_reward, robby.wall_hits, robby.good_pick_ups, robby.bad_pick_ups


if __name__ == "__main__":
    row_count = 10
    col_count = 10
    action_count = 5
    episodes = 5000
    actions_per_episode = 200
    # epsilon starts at 1.1 because its decremented on the first pass
    epsilon = 1.1
    learning_rate = .2
    discount_rate = .25
    can_probability = .5
    # initalize robby
    robby = Robby(learning_rate, discount_rate, show_move=False)
    training_rewards = []
    training_reward_varience = []
    training_wall_hits = []
    training_good_pick_ups = []
    training_bad_pickups = []
    # train robby
    for episode_index in range(episodes):
        # decrement epsilon
        if episode_index % 50 == 0 and epsilon > 0.1:
            epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            training_rewards.append(episode_value)
            training_reward_varience.append(episode_value ** 2)
            training_wall_hits.append(wall_hits)
            training_good_pick_ups.append(good_pick_ups)
            training_bad_pickups.append(bad_pickups)
    print "discount_rate= " + str(discount_rate)
    print "training rewards " + str(training_rewards)
    print "training rewards sum " + str(sum(training_rewards))
    print "training rewards average " + str(sum(training_rewards)/len(training_rewards))
    print "training rewards standard deviation " + str(math.sqrt(sum(training_reward_varience)/len(training_rewards)))
    print "wall hits: " + str(training_wall_hits)
    print "total wall hits: " + str(sum(training_wall_hits))
    print "good pickups: " + str(training_good_pick_ups)
    print "total good pickups: " + str(sum(training_good_pick_ups))
    print "bad pickups: " + str(training_bad_pickups)
    print "total bad pickups: " + str(sum(training_bad_pickups))
    print
    print
    np.save("training_rewards_discount_rate=" + str(discount_rate), np.array(training_rewards))

    # test robby
    epsilon = 0
    robby.training = False
    testing_rewards = []
    testing_reward_varience = []
    testing_wall_hits = []
    testing_good_pick_ups = []
    testing_bad_pick_ups = []
    # robby.show_move = True
    for episode_index in range(episodes):
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            testing_rewards.append(episode_value)
            testing_reward_varience.append(episode_value ** 2)
            testing_wall_hits.append(wall_hits)
            testing_good_pick_ups.append(good_pick_ups)
            testing_bad_pick_ups.append(bad_pickups)
    print "testing rewards" + str(testing_rewards)
    print "testing rewards sum " + str(sum(testing_rewards))
    print "testing rewards average " + str(sum(testing_rewards)/len(testing_rewards))
    print "testing rewards standard deviation " + str(math.sqrt(sum(testing_reward_varience)/len(testing_rewards)))
    print "wall hits: " + str(testing_wall_hits)
    print "total wall hits: " + str(sum(testing_wall_hits))
    print "good pickups: " + str(testing_good_pick_ups)
    print "total good pickups: " + str(sum(testing_good_pick_ups))
    print "bad pickups: " + str(testing_bad_pick_ups)
    print "total bad pickups: " + str(sum(testing_bad_pick_ups))
    np.save("testing_rewards_discount_rate=" + str(discount_rate), np.array(testing_rewards))

    epsilon = 1.1
    discount_rate = .50
    # initalize robby
    robby = Robby(learning_rate, discount_rate, show_move=False)
    training_rewards = []
    training_reward_varience = []
    training_wall_hits = []
    training_good_pick_ups = []
    training_bad_pickups = []
    # train robby
    for episode_index in range(episodes):
        # decrement epsilon
        if episode_index % 50 == 0 and epsilon > 0.1:
            epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            training_rewards.append(episode_value)
            training_reward_varience.append(episode_value ** 2)
            training_wall_hits.append(wall_hits)
            training_good_pick_ups.append(good_pick_ups)
            training_bad_pickups.append(bad_pickups)
    print "learning rate = " + str(discount_rate)
    print "training rewards " + str(training_rewards)
    print "training rewards sum " + str(sum(training_rewards))
    print "training rewards average " + str(sum(training_rewards)/len(training_rewards))
    print "training rewards standard deviation " + str(math.sqrt(sum(training_reward_varience)/len(training_rewards)))
    print "wall hits: " + str(training_wall_hits)
    print "total wall hits: " + str(sum(training_wall_hits))
    print "good pickups: " + str(training_good_pick_ups)
    print "total good pickups: " + str(sum(training_good_pick_ups))
    print "bad pickups: " + str(training_bad_pickups)
    print "total bad pickups: " + str(sum(training_bad_pickups))
    print
    print
    np.save("training_rewards_discount_rate=" + str(discount_rate), np.array(training_rewards))

    # test robby
    epsilon = 0
    robby.training = False
    testing_rewards = []
    testing_reward_varience = []
    testing_wall_hits = []
    testing_good_pick_ups = []
    testing_bad_pick_ups = []
    # robby.show_move = True
    for episode_index in range(episodes):
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            testing_rewards.append(episode_value)
            testing_reward_varience.append(episode_value ** 2)
            testing_wall_hits.append(wall_hits)
            testing_good_pick_ups.append(good_pick_ups)
            testing_bad_pick_ups.append(bad_pickups)
    print "testing rewards" + str(testing_rewards)
    print "testing rewards sum " + str(sum(testing_rewards))
    print "testing rewards average " + str(sum(testing_rewards)/len(testing_rewards))
    print "testing rewards standard deviation " + str(math.sqrt(sum(testing_reward_varience)/len(testing_rewards)))
    print "wall hits: " + str(testing_wall_hits)
    print "total wall hits: " + str(sum(testing_wall_hits))
    print "good pickups: " + str(testing_good_pick_ups)
    print "total good pickups: " + str(sum(testing_good_pick_ups))
    print "bad pickups: " + str(testing_bad_pick_ups)
    print "total bad pickups: " + str(sum(testing_bad_pick_ups))
    np.save("testing_rewards_discount_rate=" + str(discount_rate), np.array(testing_rewards))

    epsilon = 1.1
    discount_rate = .75
    # initalize robby
    robby = Robby(learning_rate, discount_rate, show_move=False)
    training_rewards = []
    training_reward_varience = []
    training_wall_hits = []
    training_good_pick_ups = []
    training_bad_pickups = []
    # train robby
    for episode_index in range(episodes):
        # decrement epsilon
        if episode_index % 50 == 0 and epsilon > 0.1:
            epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            training_rewards.append(episode_value)
            training_reward_varience.append(episode_value ** 2)
            training_wall_hits.append(wall_hits)
            training_good_pick_ups.append(good_pick_ups)
            training_bad_pickups.append(bad_pickups)
    print "learning rate = " + str(discount_rate)
    print "training rewards " + str(training_rewards)
    print "training rewards sum " + str(sum(training_rewards))
    print "training rewards average " + str(sum(training_rewards)/len(training_rewards))
    print "training rewards standard deviation " + str(math.sqrt(sum(training_reward_varience)/len(training_rewards)))
    print "wall hits: " + str(training_wall_hits)
    print "total wall hits: " + str(sum(training_wall_hits))
    print "good pickups: " + str(training_good_pick_ups)
    print "total good pickups: " + str(sum(training_good_pick_ups))
    print "bad pickups: " + str(training_bad_pickups)
    print "total bad pickups: " + str(sum(training_bad_pickups))
    print
    print
    np.save("training_rewards_discount_rate=" + str(discount_rate), np.array(training_rewards))

    # test robby
    epsilon = 0
    robby.training = False
    testing_rewards = []
    testing_reward_varience = []
    testing_wall_hits = []
    testing_good_pick_ups = []
    testing_bad_pick_ups = []
    # robby.show_move = True
    for episode_index in range(episodes):
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            testing_rewards.append(episode_value)
            testing_reward_varience.append(episode_value ** 2)
            testing_wall_hits.append(wall_hits)
            testing_good_pick_ups.append(good_pick_ups)
            testing_bad_pick_ups.append(bad_pickups)
    print "testing rewards" + str(testing_rewards)
    print "testing rewards sum " + str(sum(testing_rewards))
    print "testing rewards average " + str(sum(testing_rewards)/len(testing_rewards))
    print "testing rewards standard deviation " + str(math.sqrt(sum(testing_reward_varience)/len(testing_rewards)))
    print "wall hits: " + str(testing_wall_hits)
    print "total wall hits: " + str(sum(testing_wall_hits))
    print "good pickups: " + str(testing_good_pick_ups)
    print "total good pickups: " + str(sum(testing_good_pick_ups))
    print "bad pickups: " + str(testing_bad_pick_ups)
    print "total bad pickups: " + str(sum(testing_bad_pick_ups))
    np.save("testing_rewards_discount_rate=" + str(discount_rate), np.array(testing_rewards))

    epsilon = 1.1
    discount_rate = 1
    # initalize robby
    robby = Robby(learning_rate, discount_rate, show_move=False)
    training_rewards = []
    training_reward_varience = []
    training_wall_hits = []
    training_good_pick_ups = []
    training_bad_pickups = []
    # train robby
    for episode_index in range(episodes):
        # decrement epsilon
        if episode_index % 50 == 0 and epsilon > 0.1:
            epsilon -= .1
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            training_rewards.append(episode_value)
            training_reward_varience.append(episode_value ** 2)
            training_wall_hits.append(wall_hits)
            training_good_pick_ups.append(good_pick_ups)
            training_bad_pickups.append(bad_pickups)
    print "learning rate = " + str(discount_rate)
    print "training rewards " + str(training_rewards)
    print "training rewards sum " + str(sum(training_rewards))
    print "training rewards average " + str(sum(training_rewards)/len(training_rewards))
    print "training rewards standard deviation " + str(math.sqrt(sum(training_reward_varience)/len(training_rewards)))
    print "wall hits: " + str(training_wall_hits)
    print "total wall hits: " + str(sum(training_wall_hits))
    print "good pickups: " + str(training_good_pick_ups)
    print "total good pickups: " + str(sum(training_good_pick_ups))
    print "bad pickups: " + str(training_bad_pickups)
    print "total bad pickups: " + str(sum(training_bad_pickups))
    print
    print
    np.save("training_rewards_discount_rate=" + str(discount_rate), np.array(training_rewards))

    # test robby
    epsilon = 0
    robby.training = False
    testing_rewards = []
    testing_reward_varience = []
    testing_wall_hits = []
    testing_good_pick_ups = []
    testing_bad_pick_ups = []
    # robby.show_move = True
    for episode_index in range(episodes):
        # reset board and place robby
        board = Board(row_count, col_count, can_probability)
        starting_postition = get_starting_position(row_count, col_count)
        robby.new_episode(board, starting_postition, epsilon)
        episode_value, wall_hits, good_pick_ups, bad_pickups = do_episode(actions_per_episode, epsilon)
        if episode_index % 100 == 0:
            testing_rewards.append(episode_value)
            testing_reward_varience.append(episode_value ** 2)
            testing_wall_hits.append(wall_hits)
            testing_good_pick_ups.append(good_pick_ups)
            testing_bad_pick_ups.append(bad_pickups)
    print "testing rewards" + str(testing_rewards)
    print "testing rewards sum " + str(sum(testing_rewards))
    print "testing rewards average " + str(sum(testing_rewards)/len(testing_rewards))
    print "testing rewards standard deviation " + str(math.sqrt(sum(testing_reward_varience)/len(testing_rewards)))
    print "wall hits: " + str(testing_wall_hits)
    print "total wall hits: " + str(sum(testing_wall_hits))
    print "good pickups: " + str(testing_good_pick_ups)
    print "total good pickups: " + str(sum(testing_good_pick_ups))
    print "bad pickups: " + str(testing_bad_pick_ups)
    print "total bad pickups: " + str(sum(testing_bad_pick_ups))
    np.save("testing_rewards_discount_rate=" + str(discount_rate), np.array(testing_rewards))
