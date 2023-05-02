import torch
import random
import numpy as np
import os
import sys
from collections import deque  # Returns a new deque object initialized left-to-right (using append())
                               # with data from iterable. If iterable is not specified, the new deque is empty.
import time
from datetime import datetime
from snakeAI import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plots import plot
from more_plots import plot_hit

#DEVICE = torch.device('mps')     # 'cuda' if torch.cuda.is_available() else 'cpu'

"""
tar 80 - 100 games innan bra spelstrategi, i början vet den bara om miljön, gränserna.
Då går den efter maten och undviker gränserna
"""

MAX_MEMORY = 10000000     # original 100000
BATCH_SIZE = 10000      # original 1000
LR = 0.001     # original 0.001,  testat 0.01 och 0.0001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness original 0
        self.gamma = 0.7  # original 0.9 discount rate ska vara mindre än 1
        self.memory = deque(maxlen=MAX_MEMORY)  # automatically call popleft() if memory is full and remove 1
        self.model = Linear_QNet(11, 256, 3) #.to(DEVICE)  # original 11, 256, 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]  # get first item in list which is the head
        # create points next to head to check if close to boundary
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,      # food left
            game.food.x > game.head.x,      # food right
            game.food.y < game.head.y,      # food up
            game.food.y > game.head.y       # food down
        ]
        return np.array(state, dtype=int)   # dtype=int convert true/false to 1/0

    def remember(self, state, action, reward, next_state, done):        # done = game over state
        self.memory.append((state, action, reward, next_state, done))   # popleft if MAX_MEMORY is reached, save as 1 tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:  # grab batch_size of memory -> 1000 samples from memory
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples get 1000 random choosen samples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # * wildcard make it include all the files found within
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # instead of zip:
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):  # train for only 1 game state
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # ju mer tränad vår model blir vill vi minska på random moves alltså exploration och ha mer exploitation
        self.epsilon = 80 - self.n_games  # original: 80
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # the smaller epsilon get the smaller numbers will pass, och vid negativ så blir det inga random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # raw [5.0, 2.9, 0.9] max på dessa
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_ack_collision_itself = []      # my add
    plot_ack_collision_boundary = []    # my add
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        game_start_time = time.time()
        # get old state
        state_old = agent.get_state(game)

        # get move on current old state
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory for one state
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)  # done = game over

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            game_end_time = time.time()
            agent.train_long_memory()  # replay memory experience memory

            if score > record:
                record = score
                agent.model.save()

            game_time = np.round(game_end_time - game_start_time, decimals=8)
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = np.round(total_score / agent.n_games, decimals=2)

            hit_itself = game.collision_itself              # my add
            plot_ack_collision_itself.append(hit_itself)    # my add

            hit_boundary = game.collision_boundary              # my add
            plot_ack_collision_boundary.append(hit_boundary)    # my add
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            #plot_hit(plot_ack_collision_itself, plot_ack_collision_boundary)  # my add  KOMMENTERA BORT i plot hit boundary

            def write_to_file(target_path, filename):
                file_name_measure = os.path.join(target_path, filename)
                data_measure = open(file_name_measure, 'a+')
                data_measure.write(f"Date {datetime.today().isoformat()} Game {agent.n_games} Score {score} "
                                   f"Mean_score {mean_score} Record {record} Gametime {game_time}"
                                   f" hit_itself {game.collision_itself} hit_boundary {game.collision_boundary}\n")


            # my addings
            target_path = 'data_measures'
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            if agent.n_games <= 10000:
                filename = 'data_measure_80_gamma_07_21_april_1'
                write_to_file(target_path, filename)
            if 10000 < agent.n_games <= 20000:
                filename = 'data_measure_80_gamma_07_21_april_2'
                write_to_file(target_path, filename)
            if 20000 < agent.n_games <= 30000:
                filename = 'data_measure_80_gamma_07_21_april_3'
                write_to_file(target_path, filename)

                """file_name_measure = os.path.join(target_path, filename)
                data_measure = open(file_name_measure, 'a+')
                data_measure.write(f"Date {datetime.today().isoformat()} Game {agent.n_games} Score {score} "
                                   f"Mean_score {mean_score} Record {record} Gametime {game_time}"
                                   f" hit_itself {game.collision_itself} hit_boundary {game.collision_boundary}\n")

                data_measure.close()"""

            if agent.n_games == 30000:   # my add to control number of games to run
                sys.exit()


if __name__ == '__main__':
    train()
