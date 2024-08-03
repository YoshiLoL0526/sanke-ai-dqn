from game import Game
from agent import Agent
# import numpy as np
# from collections import deque


# Define constants
GAME_SIZE = 8

# Init environment and agent
env = Game(GAME_SIZE)
agent = Agent(5, 4)

# Some statistics...
max_score = 0
best_mean = 0
last_mean = 0
score = []

step = 0
for i in range(100000):

    # Reset the environment
    env.reset()
    agent.clear_memory_buffer()

    # Init last actions array
    # _last_states = deque([])

    done = False
    while not done:
        # Get current _state of environment
        _state = env.get_world()
        # _last_states.append(_state)
        # state = np.array(_last_states)
        state = _state

        # Select and do an actions
        action = agent.select_action(state)
        done, reward = env.do_action(action)

        # Get the new _state
        _new_state = env.get_world()
        # new_state = _last_states.copy()
        # new_state.append(_new_state)
        new_state = _new_state

        # Save transition into the memory
        agent.remember(state, action, new_state, reward, done)

        # Optimize model
        if step % agent.n_step == 0:
            agent.optimize_model()

        # Update target network
        if step % 2500 == 0:
            agent.update_target_network()

        # Increase step counter
        step += 1

    # Add score of episode
    score.append(env.get_score())

    # Save max score
    if env.get_score() > max_score:
        max_score = env.get_score()

    # Print mean_score score every 100 episodes
    if (i+1) % 100 == 0:
        episode_best = max(score)
        mean_score = sum(score) / len(score)
        print('Episode {}: Mean Score: {}, Episode best: {}, Highscore: {}, EPS: {}'.format(i+1, mean_score, episode_best, max_score, round(agent.epsilon, 2)))
        print('LR: ', agent.lr)

        # Save model
        if mean_score > best_mean:
            agent.save_model()
            best_mean = mean_score

        if mean_score < last_mean:
            # Reduce learning rate if underfitting
            agent.reduce_learning_rate()

        last_mean = mean_score

        score.clear()
