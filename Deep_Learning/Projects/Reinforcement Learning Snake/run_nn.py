import pygame
import numpy as np
from tensorflow import keras
from game_nn import SnakeGameAI
import sys


MODEL_PATH = "saved_models/model_64_epoch_100.keras"

nn_model = keras.models.load_model(MODEL_PATH)

nn_model.build(input_shape=(None, 11))

layer_outputs = [layer.output for layer in nn_model.layers if 'dense' in layer.name]
activation_model = keras.models.Model(inputs=nn_model.inputs[0], outputs=layer_outputs)

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

def play_snake(model):
    """Funkcja do uruchomienia gry węża z agentem."""
    pygame.init()
    game = SnakeGameAI(model)
    agent = SnakeAgent(model)

    state = game._get_state()
    clock = pygame.time.Clock()

    while True:
        action = agent.get_action(state)

        state_input = np.array(state).reshape(1, -1)
        dense_activations = activation_model.predict(state_input, verbose=0)
        activations = [state_input] + list(dense_activations)

        state, reward, done, score = game.play_step(action, activations)

        if done:
            print(f"Game Over! Score: {score}")
            pygame.quit()
            sys.exit()

        clock.tick(30)

if __name__ == "__main__":
    play_snake(nn_model)
