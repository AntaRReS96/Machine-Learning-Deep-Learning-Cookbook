GAME_SPEED = 100

SIZE_FACTOR = 2

BLOCK_SIZE = 20 * SIZE_FACTOR

WIN_WIDTH = 400 * SIZE_FACTOR
WIN_HEIGHT = 400 * SIZE_FACTOR

WIN_NN_WIDTH = 800 * SIZE_FACTOR
WIN_NN_HEIGHT = 400 * SIZE_FACTOR
DISPLAY_SPLIT = WIN_NN_WIDTH // 2

SNAKE_HEAD_COLOR = (0, 80, 0) 
SNAKE_COLOR = (0, 155, 0)    
BLACK = (0, 0, 0)
GRAY = (20, 20, 20)
RED = (255, 0, 0)

NEURON_COLOR = (0, 0, 255)
WEIGHT_POS_COLOR = (0, 255, 0)
WEIGHT_NEG_COLOR = (255, 0, 0)

# Ulepszone parametry treningu
MAX_MEMORY = 100_000
BATCH_SIZE = 32  # Zmniejszony batch size dla lepszej stabilności
LR = 0.001  # Zmniejszony learning rate dla stabilności

# Szybki harmonogram epsilon - 90% wiedzy po 100 grach
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95

# Alternatywnie: harmonogram liniowy
USE_LINEAR_EPSILON = True  # Przełącznik na liniowy scheduler
EPSILON_LINEAR_STEPS = 50  # Jeszcze szybszy: 90% wiedzy po 50 grach!

# Gamma dla discount factor
GAMMA = 0.99  # Wyższa wartość dla lepszego długoterminowego planowania

# Target network update frequency
TARGET_UPDATE_FREQ = 500  # Częstsze aktualizacje dla szybszego uczenia

EPOCHS = 2000  # Więcej epok dla lepszego treningu

SAVE_CHECKPOINTS = [10, 20, 25, 50, 100, 200, 500] + list(range(500, EPOCHS+1, 100))
SAVE_DIR = "saved_models"
