from collections import deque
from snake_config import MAX_MEMORY, BATCH_SIZE, EPOCHS, SAVE_CHECKPOINTS, SAVE_DIR, LR, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, GAMMA, TARGET_UPDATE_FREQ, USE_LINEAR_EPSILON, EPSILON_LINEAR_STEPS
from snake_game import SnakeGameAI
from snake_model import make_model
import numpy as np
import tensorflow as tf
import os
import datetime
import logging

# Ukryj komunikaty TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorBoard setup
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

try:
    summary_writer = tf.summary.create_file_writer(log_dir)
    TENSORBOARD_AVAILABLE = True
except Exception:
    summary_writer = None
    TENSORBOARD_AVAILABLE = False


os.makedirs(SAVE_DIR, exist_ok=True)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = make_model(input_shape=[16], hidden_size=64, output_size=3)
        self.target_model = make_model(input_shape=[16], hidden_size=64, output_size=3)
        self.model.compile(optimizer='adam', loss='mse')
        self.target_model.compile(optimizer='adam', loss='mse')
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.target_update_counter = 0
        self.recent_scores = deque(maxlen=10)  # Ostatnie 10 wyników do średniej

    def _epsilon_greedy_policy(self, state):
        new_action = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 3)
            new_action[action_choice] = 1
        else:
            Q_values = self.model.predict(state[np.newaxis], verbose=0)
            action_choice = np.argmax(Q_values[0])
            new_action[action_choice] = 1

        return new_action

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state):
        action = self._epsilon_greedy_policy(state)
        next_state, reward, done, info = env.play_step(action, self.epsilon, self.n_games)
        self.memory.append((state, action, reward, next_state, done))
        return next_state, action, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        
        with tf.GradientTape() as tape:
            # Current Q-values from main network
            current_Q_values = self.model(states, training=True)
            
            # Double DQN: Use main network to select actions, target network to evaluate
            next_Q_values_main = self.model(next_states, training=False)
            next_actions = tf.argmax(next_Q_values_main, axis=1)
            
            next_Q_values_target = self.target_model(next_states, training=False)
            next_actions_indices = tf.stack([tf.range(tf.shape(next_actions)[0], dtype=tf.int32), 
                                           tf.cast(next_actions, tf.int32)], axis=1)
            max_next_Q_values = tf.gather_nd(next_Q_values_target, next_actions_indices)
            
            # Target Q-values using Double DQN
            target_Q_values = rewards + self.gamma * max_next_Q_values * (1 - dones)
            
            # Q-values for taken actions
            action_indices = tf.cast(tf.argmax(actions, axis=1), tf.int32)
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            indices = tf.stack([batch_indices, action_indices], axis=1)
            current_Q_action = tf.gather_nd(current_Q_values, indices)
            
            # Loss
            loss = tf.reduce_mean(tf.square(target_Q_values - current_Q_action))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        return loss.numpy()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_model.set_weights(self.model.get_weights())
        print(f"Target network updated at step {self.target_update_counter}")

    def save_model(self, epoch):
        model_path = os.path.join(SAVE_DIR, f"aaasmol_model_64_epoch_{epoch}.h5")
        self.model.save(model_path, save_format="h5")
        print(f"Model saved at epoch {epoch}: {model_path}")

    def update_epsilon(self):
        """Aktualizuje epsilon według wybranego harmonogramu"""
        if USE_LINEAR_EPSILON:
            # Liniowy scheduler: epsilon = 0.1 po 100 grach, potem 0.01
            if self.n_games <= EPSILON_LINEAR_STEPS:
                # Liniowy spadek z 1.0 do 0.1 w pierwszych 100 grach
                target_epsilon = 0.1
                progress = self.n_games / EPSILON_LINEAR_STEPS
                self.epsilon = EPSILON_START - (EPSILON_START - target_epsilon) * progress
            else:
                # Po 100 grach: dalszy spadek do minimum
                remaining_games = self.n_games - EPSILON_LINEAR_STEPS
                if remaining_games < 100:  # Kolejne 100 gier na osiągnięcie minimum
                    progress = remaining_games / 100
                    self.epsilon = 0.1 - (0.1 - self.epsilon_min) * progress
                else:
                    self.epsilon = self.epsilon_min
        else:
            # Eksponencjalny scheduler (oryginalny)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min
        
        # Upewnij się, że epsilon nie spadnie poniżej minimum
        self.epsilon = max(self.epsilon, self.epsilon_min)

if __name__ == "__main__":
    agent = Agent()
    env = SnakeGameAI()
    env.reset()

    state = env._get_state()

    while agent.n_games < EPOCHS:
        next_state, action, reward, done, info = agent.play_one_step(env, state)
        state = next_state

        if done:
            agent.n_games += 1
            agent.recent_scores.append(info)
            
            # NAPRAWA: Redukuj epsilon raz na grę, nie za każdym krokiem
            agent.update_epsilon()
            
            # Oblicz średnią z ostatnich 10 gier
            avg_score = np.mean(agent.recent_scores) if agent.recent_scores else 0
            
            print(f"Game: {agent.n_games:4d} | Score: {info:2d} | Avg(10): {avg_score:.1f} | Epsilon: {agent.epsilon:.3f} | 🍎 {info}")
            
            if TENSORBOARD_AVAILABLE and summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('Score', info, step=agent.n_games)
                    tf.summary.scalar('Average_Score_10', avg_score, step=agent.n_games)
                    tf.summary.scalar('Epsilon', agent.epsilon, step=agent.n_games)
            
            if agent.n_games in SAVE_CHECKPOINTS:
                agent.save_model(agent.n_games)

            env.reset()

        if len(agent.memory) > BATCH_SIZE:
            loss = agent.training_step(BATCH_SIZE)
            
            if TENSORBOARD_AVAILABLE and summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=agent.n_games)        

    agent.save_model("final")