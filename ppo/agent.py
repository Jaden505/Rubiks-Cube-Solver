import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Activation, Dropout
import numpy as np
from keras.initializers.initializers_v2 import GlorotUniform

class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, clip_ratio=0.2, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_actor(self):
        states = Input(shape=(self.state_dim,))
        x = Flatten()(states)
        x = Dense(256, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        probs = Dense(self.action_dim, activation='softmax')(x)

        return Model(inputs=states, outputs=probs)

    def build_critic(self):
        states = Input(shape=(self.state_dim,))
        x = Flatten()(states)
        x = Dense(256, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Dropout(0.2)(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Dropout(0.2)(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        values = Dense(1)(x)

        return Model(inputs=states, outputs=values)

    def get_action(self, state):
        probabilities = self.actor.predict(state, verbose=0)
        action = np.random.choice(self.action_dim, p=probabilities[0])
        return action

    def train(self, states, actions, rewards, next_states, dones):
        returns, advantages = self.get_gae(rewards, states, next_states, dones)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)
            old_probs = probs.numpy()
            old_probs = old_probs + 1e-10
            ratios = tf.cast(probs / old_probs, tf.float32)
            advantages = tf.cast(tf.expand_dims(advantages, axis=1), tf.float32)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            values = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        print(f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

        grads_actor = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads_critic = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

    def get_gae(self, rewards, states, next_states, dones):
        """Calculate the Generalized Advantage Estimation (GAE)"""
        values = self.critic(states).numpy().squeeze()
        next_values = self.critic(next_states).numpy().squeeze()

        returns = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(0, len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.99 * gae * (1 - dones[t])
            returns[t] = gae + values[t]

        advantages = returns - values
        return returns, advantages
