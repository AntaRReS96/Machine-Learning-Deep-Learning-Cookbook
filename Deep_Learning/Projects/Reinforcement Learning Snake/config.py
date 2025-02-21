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

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

EPOCHS = 1000

SAVE_CHECKPOINTS = [10, 20, 25] + list(range(50, EPOCHS+1, 25))
SAVE_DIR = "saved_models"