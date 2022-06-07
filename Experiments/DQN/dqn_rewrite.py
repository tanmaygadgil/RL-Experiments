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
## Hardcoded values

# EPISODES = 900
EPSILON = 1
EPSILON_DECAY = 0.99
SAMPLE_BATCH = 256
GAMMA = 0.95
MIN_EPSILON = 0.01
UPDATE_AFTER_ACTIONS = 5
UPDATE_TARGET_NETWORK = 1000
RENDER_SAMPLING_INTERVAL = 50

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
        self.actions = np.zeros(buffer_size, dtype=np.int64)
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

        
class Model(keras.Model):
    def __init__(self, action_size):
        super(Model, self).__init__()
        self.layer1 = keras.layers.Dense(512, activation = 'relu')
        self.layer2 = keras.layers.Dense(512, activation = 'relu')
        # self.layer3 = keras.layers.Dense(64, activation = 'relu')
        
        self.layer4 = keras.layers.Dense(action_size, activation = 'linear')
        
    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.layer4(x)
        
        return x

def dqn(EPSILON, env):
    
    try:
        frames = 0
        model = Model(env.action_space.n)
        target = Model(env.action_space.n)
        optimizer = keras.optimizers.Adam(lr = 0.001)
        # loss_fn = tf.keras.losses.Huber()
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # model.compile(loss=tf.keras.losses.MeanSquaredError(), 
        #               optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0), 
        #               metrics = tf.keras.metrics.CategoricalAccuracy())
        replay = ExperienceReplay(100000, env)
        best_reward = -float('inf')
        scores = []
        renders = []
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
                    
                state_p, reward, done, _ = env.step(action=action)
                
                #Add to buffer
                replay.add(state, action, reward, state_p, done) 
                if frames % UPDATE_AFTER_ACTIONS == 0:    
                    #Sample from experience buffer
                    states, actions, rewards, states_p, dones = replay.sample()
                    
                    #Convert to tensorflow
                    # states = tf.convert_to_tensor(states, dtype = np.float32)
                    # states_p = tf.convert_to_tensor(states_p, dtype = np.float32)
                    dones = tf.convert_to_tensor(dones, dtype = np.float32)
                    rewards = tf.convert_to_tensor(rewards, dtype = np.float32)
                    
                    #Calculate targets
                    q_vals_target = target.predict(states_p, verbose = False)
                    #q_vals_target = tf.argmax(q_vals_new, axis = 1)
                    q_vals_new = model.predict(states_p, verbose = False)
                    taken_actions = np.argmax(q_vals_new, axis = 1)
                    r = np.arange(len(q_vals_new))
                    # q_target = rewards + GAMMA * q_vals_new * (1 - dones)
                    q_target = rewards + GAMMA * q_vals_target[r, taken_actions] * (1 - dones)
                    
                    #Create mask to only train on qvalues of selected actions
                    mask = tf.one_hot(taken_actions, env.action_space.n)
                    
                    #Learn
                    with tf.GradientTape() as tape:
                        q_vals = model(states)
                        # tf.cast(q_vals, dtype=np.float32)
                        #Apply mask to get the qvalue of only the action
                        q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)
                        loss = loss_fn(q_target, q_action)
                        
                    #Backpropagate
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                if frames % UPDATE_TARGET_NETWORK == 0:
                    target.set_weights(model.get_weights())
                
                if episode % RENDER_SAMPLING_INTERVAL == 0:
                    renders.append(env.render(mode="rgb_array"))
                state = state_p
                score += reward
                
                # new_weights =  model.get_weights()
                # print(f"Weights:{len(weights)}")
                # try:
                #     if weights == new_weights:
                #         print("nolearn")
                # except:
                #     pass
                # weights = new_weights
                # print(state)
                frames += 1
                if done:
                    if score > best_reward:
                        best_reward = score
                    print("Episode {}  Best Reward {} Last Reward {} Epsilon {}"\
                          .format(episode, best_reward, score, EPSILON))
                    if episode % RENDER_SAMPLING_INTERVAL == 0:
                        file_str = f"{episode}-{score}-{EPSILON}.gif"
                        
                        save_frames_as_gif(renders, 
                                        folder_string, file_str)
                        
                    renders = []
                    scores.append(score)
                    break
            # if episode % RENDER_SAMPLING_INTERVAL:
            #     env.close()   
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, MIN_EPSILON)
    except Exception as e:
        print(e)
        traceback.print_exc()
        
    finally:
        env.close() 
        return scores
    
            
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="lunarlander", type=str)
    parser.add_argument("--episodes", default=500, type=int)
    args_dict = vars(parser.parse_args())
    print(args_dict)
    EPISODES = args_dict['episodes']
    if args_dict['environment'] == "lunarlander":
        env = gym.make("LunarLander-v2")
    elif args_dict['environment'] == "cartpole":
        env = gym.make("CartPole-v1")
    
    folder_string = generate_time_string()
    os.mkdir(os.path.join("results/",folder_string))
    
    scores = dqn(EPSILON,env)
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
    
    