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

EPISODES = 500
GAMMA = 0.99
MAX_STEPS = 10000
eps = np.finfo(np.float32).eps.item()

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
    rewards_history = [] 
    best_reward = -float('inf')
    all_rewards = []
    
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for t in range(MAX_STEPS):
            
                state = tf.convert_to_tensor(state)
                state = tf.reshape(state, (1, len(state)))
            
                action_probs, critic_val = model(state)
                critic_value_history.append(critic_val[0,0])
                
                action = np.random.choice(env.action_space.n, p = np.squeeze(action_probs))
                 
                action_probs_history.append(tf.math.log(action_probs[0,action]))
                state, reward, done, _ = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward
                if done:
                    all_rewards.append(episode_reward)
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    print("Episode {}  Best Reward {} Last Reward {} "\
                          .format(episode, best_reward, episode_reward))
                    break
        
        
            returns = []
            discounted_sum = 0
        
            for r in rewards_history[::-1]:
                discounted_sum = r + GAMMA * discounted_sum
                returns.insert(0, discounted_sum)
                
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            
            for log_prob, value, ret in history:
                
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )
                
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()
            
    return all_rewards
        

if __name__ == "__main__":
    
    env = gym.make("LunarLander-v2")
    model = Model(env.action_space.n)
    
    all_rewards = ac_agent(model, env)
    
    moving_ave = []
    for i in range(len(all_rewards)):
        try:
            moving_ave.append(sum(all_rewards[i - 10:i]) / 10)
        except:
            moving_ave.append(0)
    
    plt.plot(range(len(all_rewards)), all_rewards)
    plt.plot(range(len(all_rewards)), moving_ave)
    plt.show()