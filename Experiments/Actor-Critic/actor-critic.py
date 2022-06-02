## import
## This is a vanilla implementation of actor critic
##DQN prototyping

import gym
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from tensorflow import keras
import matplotlib.pyplot as plt

## Hardcoded Variables

EPISODES = 300
GAMMA = 0.99
MAX_STEPS = 10000

class Model(keras.Model):
    def __init__(self, action_size) -> None:
        super(Model, self).__init__()
        self.layer1 = keras.layers.Dense(64, activation = 'relu')
        self.layer2 = keras.layers.Dense(64, activation = 'relu')
        
        self.actor = keras.layers.Dense(action_size, activation = 'softmax')
        self.critic = keras.layers.Dense(1)
        
    def call(self,input):
        
        x = self.layer1(input)
        x = self.layer2(x)
        
        #Action Distribution
        actor = self.actor(x)
        
        #Value of rollout of rewards
        critic = self.critic(x)
        
        return actor, critic
    
def ac_agent(model, env):
    
    optimizer = tf.keras.optimizers.Adam()
    huber_loss = tf.keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    
    for episode in EPISODES:
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for t in range(MAX_STEPS):
                state = tf.convert_to_tensor(state)
                state = tf.reshape(state, (1, len(state)))
                
                action_probs, critic_val = model(state)
                 
                

if __name__ == "__main__":
    
    env = gym.make("CartPole-v1")
    model = Model(env.action_space.n)
    
    