import pygame
import numpy as np
import tensorflow as tf
import os
import logging
import sys

# Ukryj komunikaty TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from snake_game import SnakeGameAI
from snake_game_nn import SnakeGameAI as SnakeGameNN  # Import dla wizualizacji sieci

# ===========================================
# KONFIGURACJA
# ===========================================
# Wybierz tryb gry:
# 1 - Gra bez wizualizacji sieci (szybsza)
# 2 - Gra z wizualizacją sieci neuronowej
GAME_MODE = 2

# Używaj ścieżki absolutnej względem lokalizacji skryptu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "smol_model_64_epoch_200.h5")

# Sprawdź czy model istnieje
if not os.path.exists(MODEL_PATH):
    print(f"Model nie znaleziony: {MODEL_PATH}")
    print("Dostępne modele:")
    saved_models_dir = os.path.join(SCRIPT_DIR, "saved_models")
    if os.path.exists(saved_models_dir):
        for file in os.listdir(saved_models_dir):
            print(f"  {file}")
    sys.exit(1)

# Ładuj model
try:
    nn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    nn_model.compile(optimizer='adam', loss='mse')
    
    # Sprawdź architekturę modelu
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input shape: {nn_model.input_shape}")
    print(f"Output shape: {nn_model.output_shape}")
    
    # Utwórz model aktywacji dla wizualizacji
    layer_outputs = [layer.output for layer in nn_model.layers if 'dense' in layer.name]
    if layer_outputs:
        activation_model = tf.keras.models.Model(inputs=nn_model.input, outputs=layer_outputs)
    else:
        activation_model = None
        
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    sys.exit(1)

class SnakeAgent:
    """Klasa agenta, który gra w grę węża na podstawie modelu."""
    def __init__(self, model):
        self.model = model

    def get_action(self, state):
        state = np.array(state).reshape(1, -1)  # Dopasowanie kształtu do wejścia sieci
        predictions = self.model.predict(state, verbose=0)
        action = np.argmax(predictions)
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1
        return action_one_hot

def play_snake_simple(model):
    """Funkcja do uruchomienia gry węża bez wizualizacji sieci."""
    pygame.init()
    game = SnakeGameAI()
    agent = SnakeAgent(model)

    state = game._get_state()
    clock = pygame.time.Clock()

    print("Gra uruchomiona! Naciśnij ESC aby wyjść.")
    print(f"Rozmiar stanu: {len(state)} elementów")

    while True:
        # Sprawdź zdarzenia
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Pobierz akcję od agenta
        action = agent.get_action(state)

        # Wykonaj krok w grze
        state, reward, done, score = game.play_step(action)

        # Jeśli gra się skończyła, zresetuj
        if done:
            print(f"Game Over! Score: {score}")
            state = game.reset()

        clock.tick(60)

def play_snake_with_nn_visual(model):
    """Funkcja do uruchomienia gry węża z wizualizacją sieci neuronowej."""
    pygame.init()
    game = SnakeGameNN(model)  # Użyj wersji z wizualizacją sieci
    agent = SnakeAgent(model)

    state = game._get_state()
    clock = pygame.time.Clock()

    print("Gra z wizualizacją sieci uruchomiona! Naciśnij ESC aby wyjść.")
    print(f"Rozmiar stanu: {len(state)} elementów")

    while True:
        # Sprawdź zdarzenia
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Pobierz akcję od agenta
        action = agent.get_action(state)

        # Oblicz aktywacje dla wizualizacji
        if activation_model:
            state_input = np.array(state).reshape(1, -1)
            dense_activations = activation_model.predict(state_input, verbose=0)
            activations = [state_input] + list(dense_activations)
        else:
            activations = [np.array(state).reshape(1, -1)]

        # Wykonaj krok w grze z aktywacjami
        state, reward, done, score = game.play_step(action, activations)

        # Jeśli gra się skończyła, zresetuj
        if done:
            print(f"Game Over! Score: {score}")
            state = game.reset()

        clock.tick(30)  # Wolniejsze dla lepszej wizualizacji

if __name__ == "__main__":
    # Użyj konfiguracji zamiast interaktywnego wyboru
    if GAME_MODE == 2:
        print("Uruchamiam grę z wizualizacją sieci neuronowej...")
        play_snake_with_nn_visual(nn_model)
    else:
        print("Uruchamiam grę bez wizualizacji sieci...")
        play_snake_simple(nn_model)
