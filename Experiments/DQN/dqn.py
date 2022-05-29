##DQN prototyping

from tabnanny import verbose
import gym
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from tensorflow import keras


## Hardcoded values

EPISODES = 10
EPSILON = 1
EPSILON_DECAY = 0.99
SAMPLE_BATCH = 64
GAMMA = 0.99
MIN_EPSILON = 0.01

#Experience data structure

Exp = namedtuple("experience", field_names=['s', 'a', 'r','done', 's_p'])

## Experience replay class
class ExperienceReplay():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        
    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, sample_size = SAMPLE_BATCH):
        state = []
        actions = []
        rewards = []
        dones = []
        state_p = []
        ##get sample of indices
        if sample_size > len(self.buffer):
            
            for i in range(len(self.buffer)):
                s, a, r, d, s_p = self.buffer[i]
                state.append(s)
                actions.append(a)
                rewards.append(r)
                dones.append(d)
                state_p.append(s_p)
            return np.array(state), np.array(actions), np.array(rewards), np.array(dones), np.array(state_p)
        else:
            indices = np.random.choice(len(self.buffer), 
                                       sample_size, 
                                       replace=False)
            for i in indices:
                s, a, r, d, s_p = self.buffer[i]
                state.append(s)
                actions.append(a)
                rewards.append(r)
                dones.append(d)
                state_p.append(s_p)
            return np.array(state), np.array(actions), np.array(rewards), np.array(dones), np.array(state_p)
            

class Model(keras.Model):
    def __init__(self, action_size):
        super(Model, self).__init__()
        self.layer1 = keras.layers.Dense(64)
        self.layer2 = keras.layers.Dense(128)
        # self.layer3 = keras.layers.Dense(128)
        self.layer4 = keras.layers.Dense(action_size)
        
    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
def dqn(EPSILON):
    
    try:
        model = Model(env.action_space.n)

        model.compile(loss=tf.losses.CategoricalCrossentropy(), 
                      optimizer=tf.optimizers.Adam(), 
                      metrics = tf.keras.metrics.CategoricalAccuracy())
        replay = ExperienceReplay(10000)
        best_reward = 0
        scores = []
        for episode in range(EPISODES):

            state = env.reset()
            score = 0

            while True:

                ## Choose an action
                if np.random.random() > EPSILON:
                    action = np.argmax(model.predict(state.reshape(1, -1), verbose = False), axis = 1)[0]
                else:
                    action = env.action_space.sample()
                env.render()
                # print(action)
                state_p, reward, done, _ = env.step(action=action)
                replay.add(Exp(s = state, 
                               a = action,
                               r = reward,
                               done= done, 
                               s_p = state_p))
                #Learn 
                states, actions, rewards, dones, states_p = replay.sample()
                q_vals = model.predict(states, verbose = False)
                # print(f'q_vals.shape:{q_vals.shape}')
                # print(f'actions:{actions}')
                q_vals_new = np.max(model.predict(states_p, verbose = False), axis = 1)

                q_target = rewards + GAMMA * q_vals_new * (1 - dones)
                # for idx, ele in enumerate(q_target):
                #     q_vals[idx, actions[idx]] = ele
                q_vals[ np.array(range(len(actions))), actions] = q_target
                # print(q_target.shape)
                # print(states.shape)
                model.fit(states, 
                          q_vals, 
                          epochs = 1, 
                          verbose = False)

                
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
    
    env = gym.make("LunarLander-v2")
    scores = dqn(EPSILON)
    print(scores)
        