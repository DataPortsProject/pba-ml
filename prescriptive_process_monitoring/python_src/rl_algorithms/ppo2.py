import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time


def train_run(env, ppo=None, episodes=5, steps_per_epoch=128, render=False, save_name=None, load_name=None,
              target_kl=0.0):
    if load_name is not None:
        ppo.actor = keras.models.load_model(str(save_name) + "_actor")
        ppo.critic = keras.models.load_model(str(save_name) + "_critic")

    # Initialize the observation, episode return and episode length
    observation, episode_return, episode_length = env.reset(), 0, 0
    current_episodes = 0
    epoch = 0
    while current_episodes < episodes:
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):
            if render:
                env.render()

            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            logits, action = ppo.sample_action(observation)
            observation_new, reward, done, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = ppo.critic(observation)
            logprobability_t = ppo.logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            ppo.buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else ppo.critic(observation.reshape(1, -1))
                ppo.buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0
                current_episodes += 1
                if current_episodes >= episodes:
                    break

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = ppo.buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(ppo.train_policy_iterations):
            kl = ppo.train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(ppo.train_value_iterations):
            ppo.train_value_function(observation_buffer, return_buffer)

        epoch += 1
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    if save_name is not None:
        ppo.actor.save(str(save_name) + "_actor")
        ppo.critic.save(str(save_name) + "_critic")
    env.close()


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class PPO:
    def __init__(self, observation_dimensions, num_actions, steps_per_epoch=4000):
        # Hyperparameters of the PPO algorithm
        self.gamma = 1
        self.clip_ratio = 0.2
        self.policy_learning_rate = 2.5e-4
        self.value_function_learning_rate = 2.5e-4
        self.train_policy_iterations = 4
        self.train_value_iterations = 4
        self.lam = 1
        self.hidden_sizes = (64, 64)

        self.observation_dimensions = observation_dimensions
        self.num_actions = num_actions

        # Initialize the buffer
        self.buffer = Buffer(self.observation_dimensions, steps_per_epoch)

        # Initialize the actor and the critic as keras models
        self.observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        self.logits = self.mlp(self.observation_input, list(self.hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)
        self.value = tf.squeeze(
            self.mlp(self.observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=self.observation_input, outputs=self.value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

    def mlp(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
            self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


if __name__ == '__main__':
    # Initialize the environment
    environment = gym.make("CartPole-v1")
    train_run(environment, render=True)
