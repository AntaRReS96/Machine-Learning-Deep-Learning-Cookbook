import pygame
from config import BLOCK_SIZE, WIN_WIDTH, WIN_HEIGHT, SNAKE_HEAD_COLOR, SNAKE_COLOR, BLACK, GRAY, RED

class GameDisplay:
    def __init__(self, width=WIN_WIDTH, height=WIN_HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
    
    def draw_background(self):
        """Rysowanie siatki jako tła gry."""
        for row in range(0, self.height, BLOCK_SIZE):
            for col in range(0, self.width, BLOCK_SIZE):
                color = GRAY if (row // BLOCK_SIZE + col // BLOCK_SIZE) % 2 == 0 else BLACK
                pygame.draw.rect(self.display, color, (col, row, BLOCK_SIZE, BLOCK_SIZE))

    def draw_snake(self, snake):
        """Rysowanie węża na planszy."""
        pygame.draw.rect(self.display, SNAKE_HEAD_COLOR, pygame.Rect(
            snake.snake_elements[0][0], snake.snake_elements[0][1], BLOCK_SIZE, BLOCK_SIZE))
        for block in snake.snake_elements[1:]:
            pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(
                block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))

    def draw_apple(self, apple):
        """Rysowanie jabłka na planszy."""
        pygame.draw.rect(self.display, RED, pygame.Rect(
            apple.x, apple.y, BLOCK_SIZE, BLOCK_SIZE))

    def update_display(self, snake, apple):
        """Aktualizacja ekranu gry."""
        self.draw_background()
        self.draw_snake(snake)
        self.draw_apple(apple)
        pygame.display.flip()
