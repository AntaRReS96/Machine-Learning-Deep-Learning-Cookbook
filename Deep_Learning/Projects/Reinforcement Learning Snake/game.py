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


class Snake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.head = (self.x, self.y)
        self.snake_elements = [
            self.head,
            (self.head[0] - BLOCK_SIZE, self.head[1]),
            (self.head[0] - 2 * BLOCK_SIZE, self.head[1])
        ]
        self.current_direction = Direction.RIGHT

    def update(self, action):
        """
        Aktualizacja ruchu węża.
        Oczekiwany format akcji: [skręt w lewo, jazda prosto, skręt w prawo].
        """
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = directions.index(self.current_direction)
        action_idx = np.argmax(action)

        if action_idx == 0:  # skręt w lewo
            new_idx = (current_idx - 1) % 4
        elif action_idx == 2:  # skręt w prawo
            new_idx = (current_idx + 1) % 4
        else:  # jazda prosto
            new_idx = current_idx

        self.current_direction = directions[new_idx]

        new_head_x = self.head[0]
        new_head_y = self.head[1]
        if self.current_direction == Direction.RIGHT:
            new_head_x += BLOCK_SIZE
        elif self.current_direction == Direction.LEFT:
            new_head_x -= BLOCK_SIZE
        elif self.current_direction == Direction.DOWN:
            new_head_y += BLOCK_SIZE
        elif self.current_direction == Direction.UP:
            new_head_y -= BLOCK_SIZE

        self.head = (new_head_x, new_head_y)
        self.snake_elements.insert(0, self.head)

    def is_collided(self, block=None):
        """Sprawdza, czy wąż uderzył w siebie lub ścianę."""
        if block is None:
            block = self.head
        if block in self.snake_elements[1:] or block[0] < 0 or block[0] >= WIN_WIDTH or block[1] < 0 or block[1] >= WIN_HEIGHT:
            return True
        return False
    
    def get_distance_from_apple(self, apple):
        return np.sqrt((self.head[0] - apple.x)**2 + (self.head[1] - apple.y)**2)

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
        occupied_positions = {(block[0], block[1]) for block in self.game.snake.snake_elements}
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

        if self.snake.head[0] == self.apple.x and self.snake.head[1] == self.apple.y:
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
        """
        Zwraca stan gry jako wektor, gdzie uwzględnione są:
          - niebezpieczeństwo (kolizja) przy ruchu prosto, w prawo, w lewo,
          - aktualny kierunek ruchu (4 zmienne binarne),
          - pozycja jabłka względem głowy (czy jabłko jest w lewo, w prawo, powyżej, poniżej).
        """
        head = self.snake.snake_elements[0]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.snake.current_direction)
        # Określenie kierunków relatywnych
        front_dir = self.snake.current_direction
        right_dir = clock_wise[(idx + 1) % 4]
        left_dir = clock_wise[(idx - 1) % 4]

        # Funkcja pomocnicza do obliczania pozycji bloku dla danego kierunku
        def move_point(point, direction):
            x, y = point
            if direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif direction == Direction.UP:
                y -= BLOCK_SIZE
            return (x, y)

        front_block = move_point(head, front_dir)
        right_block = move_point(head, right_dir)
        left_block = move_point(head, left_dir)

        state = [
            # Zagrożenia relatywne: czy kolizja przy ruchu prosto, w prawo, w lewo?
            self.snake.is_collided(front_block),
            self.snake.is_collided(right_block),
            self.snake.is_collided(left_block),
            # Aktualny kierunek ruchu (jako wartości binarne)
            int(self.snake.current_direction == Direction.LEFT),
            int(self.snake.current_direction == Direction.RIGHT),
            int(self.snake.current_direction == Direction.UP),
            int(self.snake.current_direction == Direction.DOWN),
            # Położenie jabłka względem głowy (względem układu współrzędnych)
            int(self.apple.x < head[0]),  # jabłko jest po lewej
            int(self.apple.x > head[0]),  # jabłko jest po prawej
            int(self.apple.y < head[1]),  # jabłko jest powyżej
            int(self.apple.y > head[1])   # jabłko jest poniżej
        ]

        return np.array(state, dtype=int)
