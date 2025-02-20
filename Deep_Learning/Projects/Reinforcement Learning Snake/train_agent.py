from collections import deque
from config import MAX_MEMORY, BATCH_SIZE, EPOCHS, SAVE_CHECKPOINTS, SAVE_DIR, LR
from game import SnakeGameAI
from model import make_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import datetime


log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)


os.makedirs(SAVE_DIR, exist_ok=True)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.998
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = make_model(
            input_shape=[11], hidden_size=64, output_size=3)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

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
        next_state, reward, done, info = env.play_step(action)
        self.memory.append((state, action, reward, next_state, done))
        return next_state, action, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + self.gamma * max_next_Q_values)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states, training=True)
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()

    def save_model(self, epoch):
        model_path = os.path.join(SAVE_DIR, f"model_64_epoch_{epoch}.keras")
        self.model.save(model_path, save_format="h5")
        print(f"Model saved at epoch {epoch}: {model_path}")

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
            print(f"Game: {agent.n_games}, Score: {info}, Epsilon: {agent.epsilon:.5f}")
            
            with summary_writer.as_default():
                tf.summary.scalar('Score', info, step=agent.n_games)
                tf.summary.scalar('Epsilon', agent.epsilon, step=agent.n_games)
            
            if agent.n_games in SAVE_CHECKPOINTS:
                agent.save_model(agent.n_games)

            env.reset()

        if len(agent.memory) > BATCH_SIZE:
            loss = agent.training_step(BATCH_SIZE)
            
            with summary_writer.as_default():
                tf.summary.scalar('Loss', loss, step=agent.n_games)        

    agent.save_model("final")