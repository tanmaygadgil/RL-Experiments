##DQN prototyping
import os
from tabnanny import verbose
import gym
import numpy as np
from sqlalchemy import false
import tensorflow as tf
from collections import deque, namedtuple
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
from matplotlib import animation
import datetime
import traceback

EPSILON = 1
EPSILON_DECAY = 0.99
SAMPLE_BATCH = 256
GAMMA = 0.99
MIN_EPSILON = 0.01
UPDATE_AFTER_ACTIONS = 5
UPDATE_TARGET_NETWORK = 1000
RENDER_SAMPLING_INTERVAL = 50
LR_ACTOR = 0.001
LR_CRITIC = 0.001
TAU = 0.005
NOISE = 0.1
MEMORY_SIZE = 200000

def generate_time_string():
    string = str(datetime.datetime.now())
    return string

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(os.path.join(path , filename), writer='imagemagick', fps=60)
    
    
class ExperienceReplay:
    def __init__(self, buffer_size, env):
        self.states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, env.action_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.states_p = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_size = buffer_size
        self.index = 0
    
    def __len__(self):

        size = min(self.index, self.buffer_size)
        return size
    
    def add(self, state, action, reward, state_p, done):

        mem_index = self.index % self.buffer_size
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_p[mem_index] = state_p
        self.dones[mem_index] = done
        self.index += 1
    
    def sample(self, sample_size = 64):
        
        max_mem = min(self.index, self.buffer_size)
        indices = np.random.choice(max_mem, sample_size, replace = True)
        
        states  = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        states_p = self.states_p[indices]
        dones   = self.dones[indices]
        
        return states, actions, rewards, states_p, dones
    
class Actor(keras.Model):
    def __init__(self, num_actions) -> None:
        super(Actor, self).__init__()
        self.layer1 = keras.layers.Dense(64, activation = 'relu')
        self.layer2 = keras.layers.Dense(64, activation = 'relu')
        self.actor = keras.layers.Dense(num_actions, activation = 'linear')
        
    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        
        x = self.actor(x)
        
        return x
        
class Critic(keras.Model):
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.layer1 = keras.layers.Dense(64, activation = 'relu')
        self.layer2 = keras.layers.Dense(64, activation = 'relu')
        self.critic = keras.layers.Dense(1, activation = 'linear')
        
    def call(self, state, action):
        x = self.layer1(tf.concat([state, action], axis = 1))
        x = self.layer2(x)
        
        x = self.critic(x)
        
        return x
    
def update_network_params(actor, actor_target, critic, critic_target, tau = 1):
    
    #Updating actor
    target_weights = actor_target.weights
    weights = []
    
    for i, weight in enumerate(actor.weights):
        weights.append(weight * tau + target_weights[i] * (1-tau))
    actor_target.set_weights(weights)
    
    #Updating critic
    target_weights = critic_target.weights
    weights = []
    
    for i, weight in enumerate(critic.weights):
        weights.append(weight * tau + target_weights[i] * (1-tau))
    critic_target.set_weights(weights)
    

    
class DDPG:
    
    def __init__(self, env) -> None:

    
        #Initialize networks
        self.actor = Actor(env.action_space.shape[0])
        self.critic = Critic()
        self.actor_target = Actor(env.action_space.shape[0])
        self.critic_target = Critic()
        
        #set weights as the same for targets and models
        self.actor.compile(optimizer = tf.keras.optimizers.Adam(lr = LR_ACTOR))
        self.actor_target.compile(optimizer = tf.keras.optimizers.Adam(lr = LR_ACTOR))
        self.critic.compile(optimizer = tf.keras.optimizers.Adam(lr = LR_CRITIC))
        self.critic_target.compile(optimizer = tf.keras.optimizers.Adam(lr = LR_CRITIC))
        
        self.update_network_params(tau = 1)
        
        #Initialize replay buffer
        self.replay = ExperienceReplay(MEMORY_SIZE, env)
        
        self.n_actions = env.action_space.shape[0]
        self.min_val = env.action_space.low[0]
        self.max_val = env.action_space.high[0]
        
    def update_network_params(self, tau = 1):
    
        #Updating actor
        target_weights = self.actor_target.weights
        weights = []
        
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + target_weights[i] * (1-tau))
        self.actor_target.set_weights(weights)
        
        #Updating critic
        target_weights = self.critic_target.weights
        weights = []
        
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + target_weights[i] * (1-tau))
        self.critic_target.set_weights(weights)
        
    def choose_action(self, observation, is_train = True):
        state = tf.convert_to_tensor([observation], dtype = np.float32)
        action = self.actor(state)
        if is_train:
            action += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=NOISE)
            
        action = tf.clip_by_value(action, clip_value_min=self.min_val, clip_value_max=self.max_val)
        
        return action
    
    def add(self, state, action, reward, state_p, done):
        
        self.replay.add(state, action, reward, state_p, done)
         
    def learn(self):
        
        states, actions, rewards, states_p, dones = self.replay.sample(SAMPLE_BATCH)
        
        states = tf.convert_to_tensor(states, dtype=np.float32)
        actions = tf.convert_to_tensor(actions, dtype=np.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=np.float32)
        states_p = tf.convert_to_tensor(states_p, dtype=np.float32)
        
        #Update Critic
        with tf.GradientTape() as tape:
            action_vals = self.actor_target(states_p)
            critic_vals_new = tf.squeeze(self.critic_target(states_p, action_vals), 1)
            critic_vals = tf.squeeze(self.critic(states, action_vals), 1)
            
            critic_target = rewards + GAMMA * critic_vals_new * (1-dones)
            
            critic_loss = tf.keras.losses.MSE(critic_target, critic_vals)
        
        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_weights))
        
        #update Actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            
            actor_loss = -self.critic(states, new_actions)
            actor_loss = tf.reduce_mean(actor_loss)
            
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_weights))
        
        self.update_network_params(TAU)                     
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="lunarlander", type=str)
    parser.add_argument("--episodes", default=500, type=int)
    args_dict = vars(parser.parse_args())
    print(args_dict)
    EPISODES = args_dict['episodes']
    if args_dict['environment'] == "lunarlander":
        env = gym.make("LunarLander-v2", 
                    continuous = True)
    elif args_dict['environment'] == "cartpole":
        env = gym.make("CartPole-v1")
    
    
    agent = DDPG(env)
    best_reward = -float('inf')
    scores = []
    renders = []
    frames = 0
    
    folder_string = generate_time_string()
    folder_string = os.path.join("results/",folder_string) 
    os.mkdir(folder_string)
    
    for episode in range(EPISODES):
        state = env.reset()
        score = 0
        
        while True:
            
            action = agent.choose_action(state)
            
            state_p, reward, done, _ = env.step(action=action)
            
            agent.add(state, action, reward, state_p, done)
            
            agent.learn()
            
            score+=reward
            
            state = state_p
            
            if done:
                if score > best_reward:
                        best_reward = score
                print("Episode {}  Best Reward {} Last Reward {}"\
                          .format(episode, best_reward, score))
                if episode % RENDER_SAMPLING_INTERVAL == 0:
                    file_str = f"{episode}-{score}.gif"
                    
                    save_frames_as_gif(renders, folder_string, file_str)   
                    
                scores.append(score)
    
    
    
    plt.show()
    print(scores)
    moving_ave = []
    for i in range(len(scores)):
        try:
            moving_ave.append(sum(scores[i - 10:i]) / 10)
        except:
            moving_ave.append(0)
    
    plt.plot(range(len(scores)), scores)
    plt.plot(range(len(scores)), moving_ave)
    plt.show()
    