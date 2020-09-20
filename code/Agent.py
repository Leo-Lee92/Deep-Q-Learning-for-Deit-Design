#%%
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers
import copy
import numpy as np
import random as rd

# 가치함수 근사모델을 정의하는 함수
def build_model(state_size, action_size):
    # seed_nb = 14
    # np.random.seed(seed_nb)
    # tf.random.set_seed(seed_nb)
    learning_rate = 0.01
    # model = Sequential()
    # model.add(Dense(64, input_dim = state_size, activation = 'relu', kernel_initializer = initializers.glorot_uniform(seed = seed_nb)))
    # model.add(Dense(64, activation = 'relu', kernel_initializer = initializers.glorot_uniform(seed = seed_nb)))
    # model.add(Dense(action_size, activation = 'linear', kernel_initializer = initializers.glorot_uniform(seed = seed_nb)))
    # model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

    model = Sequential()
    model.add(Dense(64, input_dim = state_size, activation = 'sigmoid', kernel_initializer = initializers.Ones()))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'sigmoid', kernel_initializer = initializers.Ones()))
    # model.add(Dropout(0.3))
    model.add(Dense(action_size, activation = 'linear', kernel_initializer = initializers.Ones()))
    model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

    # model = Sequential()
    # model.add(Dense(64, input_dim = state_size, activation = 'relu', kernel_initializer = initializers.Ones()))
    # model.add(Dense(64, activation = 'relu', kernel_initializer = initializers.Ones()))
    # model.add(Dense(action_size, activation = 'linear', kernel_initializer = initializers.Ones()))
    # model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

    return model

def train_model(state, action, next_action, reward, reward_grad, reward_mean, next_state, epsilon, epsilon_decay, epsilon_min, discount_factor, model, reward_depth, max_reward):
    # epsilon, epsilon_decay, epsilon_min, discount_factor, model 은 전역변수
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # print("Epsilon is :", epsilon)

    # state = state
    # state = state[20:]
    state = state[:20]
    state = np.reshape(state, [1, state.shape[0]])

    # next_state = next_state
    # next_state = next_state[20:] 
    next_state = next_state[:20] 
    next_state = np.reshape(next_state, [1, next_state.shape[0]])

    greedy_index = next_action

    target_q = model.predict(state)[0]

    # if done:
    #     target_q[actions['action'] += reward

    # else:
    # print('reward obtained : ', reward)
    alpha_lr = 0.01

    print('Training Reward Gradient is :', reward)


    if reward == reward_depth:
        target_q[action] = reward
        # target_q[action] = reward
    else:
        # target_q[action] = target_q[action] + alpha_lr * (reward + discount_factor * max(model.predict(next_state)[0]) - target_q[action]) # Q-learning 

        if reward > max_reward:
            print('Reward Gradient is :', (reward_grad))
            # print('Reward Gradient is :', (reward))

            target_q[action] = ((reward) + discount_factor * max(model.predict(next_state)[0])) # Q-learning 
            # target_q[action] = target_q[action] + alpha_lr * ((reward) + discount_factor * max(model.predict(next_state)[0]) - target_q[action]) # Q-learning 
            # target_q[action] = ((reward) + discount_factor * model.predict(next_state)[0][greedy_index]) # SARSA
            # target_q[action] = ((reward) + discount_factor * max(model.predict(next_state)[0]))

            # target_q[action] = target_q[action] + alpha_lr * ((reward_grad) + (discount_factor * max(model.predict(next_state)[0]) / (target_q[action] + 0.0001))) # Q-learning 
            # target_q[action] = 0 # Q-learning 
            # target_q[action] = target_q[action] # Q-learning 
            # target_q[action] = reward_grad # Q-learning
            # target_q[action] = reward # Q-learning
             
        # elif reward == max_reward:
        #     # target_q[action] = target_q[action] + alpha_lr * ((reward_grad) + discount_factor * max(model.predict(next_state)[0]) - target_q[action]) # Q-learning 
        #     # target_q[action] = target_q[action] # Q-learning 
        #     # target_q[action] = np.mean(target_q) # Q-learning 
        #     # target_q[action] = 0 # Q-learning 
        #     target_q[action] = target_q[action] # Q-learning 

        else:
            target_q[action] = ((reward_grad) + discount_factor * max(model.predict(next_state)[0])) # Q-learning 
            # target_q[action] = target_q[action] + alpha_lr * ((reward_grad + discount_factor * max(model.predict(next_state)[0]) - target_q[action]) # Q-learning
            # target_q[action] = target_q[action] # Q-learning 
            # target_q[action] = 0 # Q-learning 
            # target_q[action] = reward_grad # Q-learning 

            # target_q[action] = target_q[action] + alpha_lr * ((reward) + (discount_factor * max(model.predict(next_state)[0]) / (target_q[action] + 0.0001))) # Q-learning 
            print('Reward Gradient is :', (reward_grad))

            
    
    print("target_q_values : ", target_q)
    target_q = np.reshape(target_q, [1, target_q.shape[0]])
 
    hist = model.fit(state, target_q, epochs = 1, verbose = 0)
    mse_loss = hist.history['loss']

    return(epsilon, mse_loss)
    # print(
    # "greedy_index : ", greedy_index,
    # "q_values_of_greedy : ", target_q[greedy_index],
    # "reward : ", reward,
    # "target_q : ", target_q
    # )

# 행위 (어느방향으로 얼만큼 갈지)를 선택하는 함수
def get_action(state_vector, actions, model, epsilon):


    # print('random_number is :', random_number)
    # print('epsilon is :', epsilon)

    # explorer
    random_number = np.random.rand()
    if random_number < epsilon:
        action = rd.randrange(len(actions))

    # exploit
    else:
        # state = state_vector
        # state = state_vector[20:] # 0~19까지의 요소 제외
        state = state_vector[:20] # 0~19까지의 요소만 활용
        state = np.reshape(state, [1, state.shape[0]]) # 14 포지션 : 길이 34
        print('State Length is :', state.shape[1])
        q_values = model.predict(state)
        # action = np.argmax(q_values)
        print('Q_VALUES :', q_values)

        action = np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    return(action, state_vector)
# %%
