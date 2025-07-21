import pygame
from snake_config import BLOCK_SIZE, WIN_WIDTH, WIN_HEIGHT, SNAKE_HEAD_COLOR, SNAKE_COLOR, BLACK, GRAY, RED

class GameDisplay:
    def __init__(self, width=WIN_WIDTH, height=WIN_HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
        
        # Inicjalizacja fontu do wyświetlania punktów
        self.font = pygame.font.Font(None, 36)
        self.score_color = (255, 255, 255)  # Biały tekst
    
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

    def draw_score(self, score):
        """Rysowanie licznika punktów w lewym górnym rogu."""
        score_text = self.font.render(f"Score: {score}", True, self.score_color)
        # Dodaj czarne tło za tekstem dla lepszej widoczności
        text_rect = score_text.get_rect()
        text_rect.topleft = (10, 10)
        
        # Rysuj czarne tło
        background_rect = text_rect.inflate(10, 5)
        pygame.draw.rect(self.display, BLACK, background_rect)
        
        # Rysuj tekst
        self.display.blit(score_text, text_rect)

    def draw_additional_info(self, score, epsilon=None, game_count=None):
        """Rysowanie dodatkowych informacji (punkty, epsilon, liczba gier)."""
        y_offset = 10
        
        # Punkty
        score_text = self.font.render(f"Score: {score}", True, self.score_color)
        score_rect = score_text.get_rect()
        score_rect.topleft = (10, y_offset)
        background_rect = score_rect.inflate(10, 5)
        pygame.draw.rect(self.display, BLACK, background_rect)
        self.display.blit(score_text, score_rect)
        
        # Epsilon (jeśli dostępny)
        if epsilon is not None:
            y_offset += 40
            epsilon_text = self.font.render(f"Epsilon: {epsilon:.3f}", True, self.score_color)
            epsilon_rect = epsilon_text.get_rect()
            epsilon_rect.topleft = (10, y_offset)
            background_rect = epsilon_rect.inflate(10, 5)
            pygame.draw.rect(self.display, BLACK, background_rect)
            self.display.blit(epsilon_text, epsilon_rect)
        
        # Liczba gier (jeśli dostępna)
        if game_count is not None:
            y_offset += 40
            game_text = self.font.render(f"Games: {game_count}", True, self.score_color)
            game_rect = game_text.get_rect()
            game_rect.topleft = (10, y_offset)
            background_rect = game_rect.inflate(10, 5)
            pygame.draw.rect(self.display, BLACK, background_rect)
            self.display.blit(game_text, game_rect)

    def update_display(self, snake, apple, score=0, epsilon=None, game_count=None):
        """Aktualizacja ekranu gry z dodatkowymi informacjami."""
        self.draw_background()
        self.draw_snake(snake)
        self.draw_apple(apple)
        self.draw_additional_info(score, epsilon, game_count)
        pygame.display.flip()
