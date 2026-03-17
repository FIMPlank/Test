# This is a sample Python script.
# Test edit to verify repo write permissions.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import transformers
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
import numpy as np
import random
from tqdm import tqdm

class GymEnvironment:
    def __init__(self, env_id, max_timesteps=120):

        self.max_timesteps = max_timesteps  # Define maximum number of timesteps the agent can balance the pole
        self.env = gym.make(env_id)  # Create the environment

    def trainDQN(self, agent):
        rew_hist, loss = self.runDQN(agent, training=True)

        # Automatically save weights of trained network
        agent.model.save_weights("tmp.h5", overwrite=True)  # Save trained weights in your working folder
        return rew_hist, loss

    def runDQN(self, agent, training=False):

        # Define two lists to save collected rewards and losses
        rew_hist = []
        loss = []

        max_no_episodes = 1000
        # TODO: Define maximum number of episodes appropriately

        target_model_update_counter = 0  # Initialize counter for target network synchronization

        # Main loop - see DQN pseudo-code in lecture
        for episode in range(max_no_episodes):

            # Initialize state and rewards for episode
            state = self.env.reset()[0].reshape(1, self.env.observation_space.shape[0])
            total_reward = 0

            done = False
            # Go through all time steps
            t = 0
            while not done and t < self.max_timesteps:  # for t in range(self.max_timesteps+1):

                # TODO: Implement appropriate action selection - action label
                action = agent.select_action(state, explore=training)

                # Execute the selected action and   observe next state + rewards
                next_state, reward, done, _, __ = self.env.step(action)  # [0]
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                if training:
                    agent.record(state, next_state, done, reward, action)
                    agent.update_weights()



                state = next_state
                total_reward += reward
                t += 1

                # TODO: Record states, rewards etc. and update network weights




            rew_hist.append(total_reward)
            # TODO: Test performance of policy every certain episode Test performance of your trained agent every 25 epsiodes

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, max_no_episodes, total_reward, agent.epsilon))

            if episode % 25 == 0 and not training:  # Test performance of your trained agent every 25 episodes
                self.testDQN(agent)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

        return rew_hist, loss

    def testDQN(self, agent):
        total_episodes = 100
        total_rewards = []

        for episode in range(total_episodes):
            state = self.env.reset()[0].reshape(1, self.env.observation_space.shape[0])
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, explore=False)

                next_state, reward, done, _, __ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print("Average test reward over {} episodes: {}".format(total_episodes, avg_reward))
        return avg_reward

class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions, load_old_model=False):

        self.state_vector_size = no_of_states
        self.action_space_size = no_of_actions

        # TODO: Define here all necessary hyperparameters
        self.gamma = 0.9  # discount rate on future rewards
        self.epsilon = 10  # exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay factor

        # Initialize the DQN neural network model
        self.model = self.nn_model(load_old_model)

        # Define the target model for stable learning
        self.target_model = self.nn_model(load_old_model=False)
        self.target_model.set_weights(self.model.get_weights())

        # Initialize the replay buffer
        self.replay_buffer = deque(maxlen=5000)

    def nn_model(self, load_old_model=False):
        if load_old_model:
            # Load the existing model from tmp.h5 file in the current working directory
            model = tf.keras.models.load_model('tmp.h5')
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, input_shape=(self.state_vector_size,), activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.action_space_size, activation='linear')
            ])

            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    # ... (the rest of the class code remains the same)


    def select_action(self, state, explore=True):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = self.model(state_tensor)

        if explore and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            return tf.argmax(q_values[0])

    def record(self, state, next_state, done, reward, action):
        self.replay_buffer.append((state, np.array(next_state), done, reward, action))

    def update_weights(self):
        if len(self.replay_buffer) < 32:  # Batch size for weight updates
            return

        batch = random.sample(self.replay_buffer, 32)
        state_batch, next_state_batch, done_batch, reward_batch, action_batch = zip(*batch)

        state_batch = np.vstack(state_batch)
        next_state_batch = np.vstack(next_state_batch)
        target = self.model.predict(state_batch)
        target_next = self.target_model.predict(next_state_batch)

        for i in range(len(batch)):
            if done_batch[i]:
                target[i][action_batch[i]] = reward_batch[i]
            else:
                target[i][action_batch[i]] = reward_batch[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(state_batch, target, epochs=1, verbose=0)

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=1000, max_timesteps=120):
        rew_hist = []

        for episode in range(max_episodes):
            state = self.env.reset().reshape(1, self.state_vector_size)
            total_reward = 0

            for t in range(max_timesteps):
                action = self.select_action(state)

                next_state, reward, done, _, __ = self.env.step(action)
                next_state = next_state.reshape(1, self.state_vector_size)

                if done:
                    reward = -10  # Penalize the agent for failing early

                self.record(state, next_state, done, reward, action)
                self.update_weights()

                state = next_state
                total_reward += reward

                if done:
                    break

            rew_hist.append(total_reward)
            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, max_episodes, total_reward, self.epsilon))

            if episode % 2 == 0:
                self.update_target_weights()

            # Decay exploration rate (epsilon)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return rew_hist


if __name__ == "__main__":
        environment = GymEnvironment('Acrobot-v1')

        total_iterations = 10

        for i in tqdm(range(total_iterations), desc="Processing"):
            # TODO: Define the dimension of the state vector and the number of actions
            state_vector_size = environment.env.observation_space.shape[0]
            action_space_size = environment.env.action_space.n

            # If load_model = 1, load an existing set of weights in current working directory
            load_model = False

            # Train the agent
            agent = DQN_Agent(state_vector_size, action_space_size) # Create agent - reinitialize for each trial
            rew_hist,loss = environment.trainDQN(agent) # Let the agent train


            # Let agent run for 100 episodes
            rew_hist,loss = environment.runDQN(agent)

            environment.testDQN(agent)







