import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)  # create a new Font object from a file


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # my add
PEACHPUFF = (255, 218, 185)  # my add
ORANGE = (255, 165, 0)  # my add

BLOCK_SIZE = 20
SPEED = 60  # original 40  # Change speed here for quicker or slower speed, kört med 60 om ej angett annat


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.collision_itself = 0
        self.collision_boundary = 0

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]  # initial snake are 3 boxes long
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over,  collision or the longer snake the more time it gets before break
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(
                self.snake):  # if iteration is to long, e.g don't find food the is_collision
            game_over = True
            reward -= 10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            self.collision_boundary += 1
            return True

        # hits itself
        if pt in self.snake[1:]:
            self.collision_itself += 1
            return True

        return False

    def _update_ui(self):
        self.display.fill(WHITE)

        for pt in self.snake:
            if pt == self.snake[0]:  # my add to see head
                pygame.draw.rect(self.display, ORANGE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  # my add ORANGE
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # my add also add else
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # straight, right, left
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # decide action to take
        if np.array_equal(action, [1, 0, 0]):  # going straight
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            next_idx = (idx + 1) % 4  # clock wise, go to RIGHT
            new_dir = clock_wise[next_idx]  # right turn right -> down -> left -> up
        else:  # [0, 0, 1]                         # left turn
            next_idx = (idx - 1) % 4  # counterclockwise, go to left
            new_dir = clock_wise[next_idx]  # left turn right -> up -> left -> up

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
