#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a simple test based on tables and reinforcement learning using gym libraries and keras to train the basic
example of cartpole


# Example based on following blog link:

https://keon.io/deep-q-learning/


"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In the reinforcement learning there is a function called Q Function, which is used to approximate the reward based on
# a state. We call it Q(s,a) where Q is a function which calculates the expected future value from state s and action a.
# We will use a neuronal network to approximate the reward based on the state.

#
#   Remember:  Q Network
#
#   * Q(s,a) where:     -> Q function which calculates the expected future value (reward).
#                       -> s. Te current state of the agent.
#                       -> a. The action to take.
#

###################### Cardpole Game

"""
CartPole is one of the simplest environments in OpenAI gym. The goal of CartPole is to balance a pole connected 
with one joint on top of a moving cart. Instead of pixel information, there are 4 kinds of information given by the s
tate, such as angle of the pole and position of the cart. 

An agent can move the cart by performing a series of actions of 0 or 1 to the cart, pushing it left or right.


"""

# Thus the action (a) can be either 0 or 1 (right or left). In gym each step (action performed), returns 4 variables
#
#   * next_state:   The information of the agent (s)
#   * reward:       The total reward obtained in the performed step
#   * done          If the game is end or not (if true, the game is ended)
#   * info          Additional information given by gym library.

# We are going to create using Keras, a Neuronal network based on Dense-NN. In image recognition we use convolutional
# neuronal networks. This is not the case of this example.
#


# Number of total episodes to run
EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        # Sequencial class creates a new NN.
        model = Sequential()
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # The state size are the 4 variables used in openAIGYM
    # next_state, reward, done, info (see above)
    state_size = env.observation_space.shape[0]
    # The total action size. In this case we have only 2 actions. 0 or 1.
    action_size = env.action_space.n
    # Loading the Agent model with the NN.
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    max_reward = 0
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                if reward > max_reward:
                    max_reward = reward
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
#     agent.save("./save/cartpole-dqn.h5")

    # After finishing, we print the max reward obtained
    print("Maximum reward obtained in the test is: "+max_reward)
