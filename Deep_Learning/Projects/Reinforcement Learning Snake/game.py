from config import BLOCK_SIZE, WIN_WIDTH, WIN_HEIGHT, GAME_SPEED
from display import GameDisplay  
import pygame
from enum import Enum
from collections import namedtuple
import random
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Block = namedtuple("Block", "x, y")


class Snake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.head = Block(self.x, self.y)
        self.snake_elements = [
            self.head,
            Block(self.head.x - BLOCK_SIZE, self.head.y),
            Block(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.current_direction = Direction.RIGHT

    def update(self, action):
        """Aktualizacja ruchu węża na podstawie akcji.
        Oczekiwany format akcji: [góra, dół, lewo, prawo] jako wektor one-hot.
        Jeśli wybrany kierunek jest przeciwny do aktualnego, wąż kontynuuje ruch w aktualnym kierunku.
        """
        
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        
        idx = np.argmax(action)
        new_direction = directions[idx]

        
        if (self.current_direction == Direction.UP and new_direction == Direction.DOWN) or \
           (self.current_direction == Direction.DOWN and new_direction == Direction.UP) or \
           (self.current_direction == Direction.LEFT and new_direction == Direction.RIGHT) or \
           (self.current_direction == Direction.RIGHT and new_direction == Direction.LEFT):
            new_direction = self.current_direction

        self.current_direction = new_direction

        
        new_head_x = self.head.x
        new_head_y = self.head.y
        if self.current_direction == Direction.RIGHT:
            new_head_x += BLOCK_SIZE
        elif self.current_direction == Direction.LEFT:
            new_head_x -= BLOCK_SIZE
        elif self.current_direction == Direction.DOWN:
            new_head_y += BLOCK_SIZE
        elif self.current_direction == Direction.UP:
            new_head_y -= BLOCK_SIZE

        self.head = Block(new_head_x, new_head_y)
        self.snake_elements.insert(0, self.head)

    def is_collided(self, block=None):
        """Sprawdza, czy wąż uderzył w siebie lub ścianę."""
        if block is None:
            block = self.head
        if block in self.snake_elements[1:] or block.x < 0 or block.x >= WIN_WIDTH or block.y < 0 or block.y >= WIN_HEIGHT:
            return True
        return False
    
    def get_distance_from_apple(self, apple):
        return np.sqrt((self.head.x - apple.x)**2 + (self.head.y - apple.y)**2)

class Apple:
    def __init__(self, game):
        self.game = game
        self.x, self.y = self.spawn_apple()

    def spawn_apple(self):
        """Losowe umiejscowienie jabłka w grze."""
        all_positions = [
            (x, y) for x in range(0, WIN_WIDTH, BLOCK_SIZE)
            for y in range(0, WIN_HEIGHT, BLOCK_SIZE)
        ]
        occupied_positions = {(block.x, block.y) for block in self.game.snake.snake_elements}
        free_positions = list(set(all_positions) - occupied_positions)

        if not free_positions:
            return -BLOCK_SIZE, -BLOCK_SIZE  

        return random.choice(free_positions)

    def change_pos(self):
        """Zmiana pozycji jabłka po zjedzeniu."""
        self.x, self.y = self.spawn_apple()

class SnakeGameAI:
    def __init__(self):
        self.width = WIN_WIDTH
        self.height = WIN_HEIGHT
        self.display = GameDisplay()  
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resetuje stan gry."""
        self.snake = Snake(self.width / 2, self.height / 2)
        self.apple = Apple(self)
        self.game_over = False
        self.score = 0

    def play_step(self, action):
        old_distance = self.snake.get_distance_from_apple(self.apple)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.snake.update(action)

        reward = 0
        if self.snake.is_collided():
            self.game_over = True
            reward -= 100
            return self._get_state(), reward, self.game_over, self.score

        if self.snake.head.x == self.apple.x and self.snake.head.y == self.apple.y:
            self.score += 1
            reward += 10
            self.apple.change_pos()
        else:
            self.snake.snake_elements.pop()

        if self.snake.get_distance_from_apple(self.apple) < old_distance:
            reward += 5
        else:
            reward -= 5

        self.display.update_display(self.snake, self.apple)
        self.clock.tick(GAME_SPEED)

        return self._get_state(), reward, self.game_over, self.score

    def _get_state(self):
        snake_head = self.snake.snake_elements[0]

        block_left = Block(snake_head.x - BLOCK_SIZE, snake_head.y)
        block_right = Block(snake_head.x + BLOCK_SIZE, snake_head.y)
        block_up = Block(snake_head.x, snake_head.y - BLOCK_SIZE)
        block_down = Block(snake_head.x, snake_head.y + BLOCK_SIZE)

        is_direction_left = self.snake.current_direction == Direction.LEFT
        is_direction_right = self.snake.current_direction == Direction.RIGHT
        is_direction_up = self.snake.current_direction == Direction.UP
        is_direction_down = self.snake.current_direction == Direction.DOWN

        state = [
            
            (is_direction_left and self.snake.is_collided(block_left)) or
            (is_direction_right and self.snake.is_collided(block_right)) or
            (is_direction_up and self.snake.is_collided(block_up)) or
            (is_direction_down and self.snake.is_collided(block_down)),

            
            (is_direction_left and self.snake.is_collided(block_up)) or
            (is_direction_right and self.snake.is_collided(block_down)) or
            (is_direction_up and self.snake.is_collided(block_right)) or
            (is_direction_down and self.snake.is_collided(block_left)),

            
            (is_direction_left and self.snake.is_collided(block_down)) or
            (is_direction_right and self.snake.is_collided(block_up)) or
            (is_direction_up and self.snake.is_collided(block_left)) or
            (is_direction_down and self.snake.is_collided(block_right)),

            
            is_direction_left,
            is_direction_right,
            is_direction_up,
            is_direction_down,

            
            self.apple.x < snake_head.x,  
            self.apple.x > snake_head.x,  
            self.apple.y < snake_head.y,  
            self.apple.y > snake_head.y  
        ]

        return np.array(state, dtype=int)
