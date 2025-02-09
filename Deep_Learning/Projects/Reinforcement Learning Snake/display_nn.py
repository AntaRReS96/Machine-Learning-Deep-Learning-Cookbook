import pygame
from config import BLOCK_SIZE, WIN_NN_WIDTH, WIN_NN_HEIGHT, DISPLAY_SPLIT, \
                   SNAKE_HEAD_COLOR, SNAKE_COLOR, BLACK, GRAY, RED, \
                   WEIGHT_POS_COLOR, WEIGHT_NEG_COLOR

class GameDisplay:
    def __init__(self, width=WIN_NN_WIDTH, height=WIN_NN_HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")

    def draw_background(self):
        """Rysowanie siatki jako tła gry."""
        self.display.fill(BLACK)
        
        # Lewa strona (sieć neuronowa)
        pygame.draw.rect(self.display, GRAY, (0, 0, DISPLAY_SPLIT, self.height))

        # Prawa strona (plansza gry)
        for row in range(0, self.height, BLOCK_SIZE):
            for col in range(DISPLAY_SPLIT, self.width, BLOCK_SIZE):
                color = GRAY if (row // BLOCK_SIZE + col // BLOCK_SIZE) % 2 == 0 else BLACK
                pygame.draw.rect(self.display, color, (col, row, BLOCK_SIZE, BLOCK_SIZE))

    def draw_snake(self, snake):
        """Rysowanie węża na planszy (przesunięcie na prawą stronę)."""
        offset_x = DISPLAY_SPLIT
        for i, block in enumerate(snake.snake_elements):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_COLOR
            pygame.draw.rect(self.display, color, pygame.Rect(block.x + offset_x, block.y, BLOCK_SIZE, BLOCK_SIZE))

    def draw_apple(self, apple):
        """Rysowanie jabłka na planszy"""
        pygame.draw.rect(self.display, RED, pygame.Rect(
            apple.x + DISPLAY_SPLIT, apple.y, BLOCK_SIZE, BLOCK_SIZE))

    def compute_neuron_positions(self, nn_model):
        """
        Oblicza pozycje neuronów dla każdej warstwy.
        Zwraca listę list z krotkami (x, y) dla każdej warstwy.
        """
        
        layer_sizes = [nn_model.input_shape[1]] + [layer.units for layer in nn_model.layers if 'dense' in layer.name]
        num_layers = len(layer_sizes)
        max_neurons = max(layer_sizes)
        
        layer_spacing = DISPLAY_SPLIT // (num_layers + 1)
        neuron_spacing = self.height // (max_neurons + 1)

        neuron_positions = []
        for i, num_neurons in enumerate(layer_sizes):
            x = (i + 1) * layer_spacing
            y_offset = (self.height - (num_neurons * neuron_spacing)) // 2

            layer_neurons = []
            for j in range(num_neurons):
                y = y_offset + j * neuron_spacing
                layer_neurons.append((x, y))
            neuron_positions.append(layer_neurons)

        return neuron_positions

    def draw_weights(self, nn_model, neuron_positions):
        """Rysowanie wag połączeń między neuronami (tylko w lewej części ekranu)"""
        weights = nn_model.get_weights()[::2]  
        
        for layer_idx, weight_matrix in enumerate(weights):
            start_neurons = neuron_positions[layer_idx]
            end_neurons = neuron_positions[layer_idx + 1]

            for i, start_pos in enumerate(start_neurons):
                for j, end_pos in enumerate(end_neurons):
                    weight_value = weight_matrix[i, j]
                    color = WEIGHT_POS_COLOR if weight_value > 0 else WEIGHT_NEG_COLOR
                    width = max(1, int(abs(weight_value) * 5))  
                    pygame.draw.line(self.display, color, start_pos, end_pos, width)

    def draw_neuron_circles(self, activations, neuron_positions, radius=5):
        """
        Rysuje neurony jako koła o podanym promieniu (domyślnie mniejszym, tj. 5).
        Kolor koła zależy od aktywacji neuronu (0 -> czarny, 1 -> biały).
        """

        for i, neurons in enumerate(neuron_positions):
            
            num_activations = activations[i].shape[1] if activations[i].ndim == 2 else activations[i].shape[0]
            for j, (x, y) in enumerate(neurons):
                if j < num_activations:
                    if activations[i].ndim == 2:
                        activation_value = activations[i][0, j]
                    else:
                        activation_value = activations[i][j]
                else:
                    activation_value = 0  
                grayscale = max(0, min(255, int(activation_value * 255)))
                neuron_color = (grayscale, grayscale, grayscale)
                pygame.draw.circle(self.display, neuron_color, (x, y), radius)

    def update_display(self, snake, apple, nn_model, activations):
        """Aktualizacja ekranu gry i rysowanie sieci neuronowej wraz z aktywacjami."""
        self.draw_background()
        self.draw_snake(snake)
        self.draw_apple(apple)
        
        neuron_positions = self.compute_neuron_positions(nn_model)
        
        self.draw_weights(nn_model, neuron_positions)
        
        self.draw_neuron_circles(activations, neuron_positions, radius=5)

        pygame.display.flip()
