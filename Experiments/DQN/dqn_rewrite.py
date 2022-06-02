##DQN prototyping

from os import replace
from tabnanny import verbose
import gym
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse

## Hardcoded values

EPISODES = 600
EPSILON = 1
EPSILON_DECAY = 0.95
SAMPLE_BATCH = 64
GAMMA = 0.95
MIN_EPSILON = 0.01

class ExperienceReplay:
    def __init__(self, buffer_size, env):
        self.states = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.states_p = np.zeros((buffer_size, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool)
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

        
class Model(keras.Model):
    def __init__(self, action_size):
        super(Model, self).__init__()
        self.layer1 = keras.layers.Dense(64, activation = 'relu')
        self.layer2 = keras.layers.Dense(64, activation = 'relu')
        # self.layer3 = keras.layers.Dense(64, activation = 'relu')
        
        self.layer4 = keras.layers.Dense(action_size)
        
    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.layer4(x)
        
        return x

def dqn(EPSILON, env):
    
    try:
        model = Model(env.action_space.n)
        optimizer = keras.optimizers.Adam()
        loss_fn = tf.keras.losses.Huber()
        # model.compile(loss=tf.keras.losses.MeanSquaredError(), 
        #               optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0), 
        #               metrics = tf.keras.metrics.CategoricalAccuracy())
        replay = ExperienceReplay(100000, env)
        best_reward = -float('inf')
        scores = []
        for episode in range(EPISODES):

            state = env.reset()
            score = 0

            while True:
                ## Choose an action
                if np.random.random() > EPSILON:
                    # action = np.argmax(model.predict(state.reshape(1, -1), verbose = False), axis = 1)[0]
                    # action = np.argmax(model.predict(state.reshape(1, -1), verbose = False))
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training = False)
                    action = tf.argmax(action_probs[0]).numpy()
                else:
                    action = env.action_space.sample()
                # env.render()
                
                state_p, reward, done, _ = env.step(action=action)
                #Add to buffer
                replay.add(state, action, reward, state_p, done) 
                #Sample from experience buffer
                states, actions, rewards, states_p, dones = replay.sample()
                
                #Convert to tensorflow
                states = tf.convert_to_tensor(states, dtype = np.float32)
                states_p = tf.convert_to_tensor(states_p, dtype = np.float32)
                dones = tf.convert_to_tensor(dones, dtype = np.float32)
                rewards = tf.convert_to_tensor(rewards, dtype = np.float32)
                
                q_vals_new = model.predict(states_p, verbose = False)
                q_vals_new = tf.reduce_max(q_vals_new, axis = 1)
                q_target = rewards + GAMMA * q_vals_new * (1-dones)
                
                #Create mask to only train on qvalues of selected actions
                mask = tf.one_hot(actions, env.action_space.n, dtype = np.float32)
                
                #Learn
                with tf.GradientTape() as tape:
                    q_vals = model(states)
                    tf.cast(q_vals, dtype=np.float32)
                    #Apply mask to get the qvalue of only the action
                    q_vals = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)
                    
                    loss = loss_fn(q_target, q_vals)
                    
                #Backpropagate
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                # q_vals = model.predict(states, verbose = False)
                # # q_vals = np.zeros((states.shape[0], env.action_space.n))
                # # print(f'q_vals.shape:{q_vals.shape}')
                # # print(f'actions:{actions}')
                # q_vals_new = np.max(model.predict(states_p, verbose = False), axis = 1)

                # q_target = rewards + GAMMA * q_vals_new * dones
                # # q_vals[ np.array(range(len(actions))), actions] = q_target
                # # print(q_target.shape)
                # q_vals[ np.array(range(len(actions))), actions] = q_target
                # # print(q_vals.shape)
                # # print(states.shape)
                # model.fit(states, 
                #           q_vals, 
                #           epochs = 1, 
                #           verbose = False)
                state = state_p
                score += reward
                # print(state)
                if done:
                    if score > best_reward:
                        best_reward = score
                    print("Episode {}  Best Reward {} Last Reward {} Epsilon {}"\
                          .format(episode, best_reward, score, EPSILON))
                    
                    scores.append(score)
                    break
            EPSILON *= EPSILON_DECAY
            
    except Exception as e:
        print(e)
        
    finally:
        return scores
    
            
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="lunarlander", type=str)
    
    args_dict = vars(parser.parse_args())
    print(args_dict)
    if args_dict['environment'] == "lunarlander":
        env = gym.make("LunarLander-v2")
    elif args_dict['environment'] == "cartpole":
        env = gym.make("CartPole-v1")
        
        
    scores = dqn(EPSILON,env)
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
    
    