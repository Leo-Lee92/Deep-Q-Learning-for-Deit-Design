# %%
import pandas as pd
import numpy as np
# from numpy.random import seed
import collections as col
import random as rd
import csv
from pathlib import Path
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras import initializers
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from numpy import linspace, exp
from scipy.special import beta, gamma

#%%
def data_preprocessing(food_df, list_of_menus_df):
    
    # (2) 데이터 통합
    ## 오전간식, 점심, 오후간식, 저녁 별 고유음식들 
    per_time = []
    for index, dataset in enumerate(list_of_menus_df):
        dataset = dataset.drop(['date'], axis = 1) # date 컬럼(axis = 1) 드롭한 데이터 프레임
        flattend_vector = dataset.values.ravel() # 데이터 프레임의 나눠진 컬럼들을 얽힘(ravel)해주기 (flatten)
        flattend_vector = [x for x in flattend_vector if str(x) != 'nan' and str(x) != ' '] # nan값과 빈값 제외해주기
        flattend_vector = set(flattend_vector) # flattend_vector에서 unique한 값만 뽑아주기
        per_time.append(flattend_vector)

    ## 시간대별 활용되는 음식 종류 (리스트)
    per_time[1] 

    ## 시간대별 활용되는 음식종류 (데이터 프레임)
    foods_by_time = pd.DataFrame(per_time).T 

    # (3) 데이터 전처리
    ## (3-1) 데이터 프레임 Wide to Long으로 melt
    list_of_melted_df = []
    for x in list_of_menus_df:
        x
        melted_df = x.melt(id_vars = 'date', var_name = 'type', value_name = 'menu').sort_values(by = ['date', 'type'], ascending = [True, True])
        melted_df = melted_df.reset_index(drop = True)
        list_of_melted_df.append(melted_df)

    ## (3-2) melt 데이터 프레임들 join
    ### 리스트에 저장된 long 데이터 프레임들을 행방향으로 concat(rbind)해주기
    concat_melted_df = pd.concat(list_of_melted_df, axis = 0, join = 'outer')
    concat_melted_df = concat_melted_df.drop(['type'], axis = 1)

    ### 각 시간대별 라벨을 추가해주기
    x_vec = []
    x_label = 0
    for x in range(len(list_of_melted_df)):    
        x_len = list_of_melted_df[x].shape[0]
        x_vec += [x_label] * x_len
        x_label += 1
    x_vec # 시간대 라벨임

    concat_melted_df['time_label'] = x_vec
    concat_melted_df = concat_melted_df.sort_values(by = ['date', 'time_label'])
    concat_melted_df = concat_melted_df.reset_index(drop = True)

    x_vec2 = list(range(14)) * int(concat_melted_df.shape[0] / 14) # 포지션임
    concat_melted_df['position'] = x_vec2

    ### 날짜가 menu에 복제된 애들 지워주기
    del_index_list = []
    for index in range(concat_melted_df.shape[0]):
        # print(index)
        if concat_melted_df.iloc[index]['date'] == concat_melted_df.iloc[index]['menu']:
            del_index_list.append(index)

    concat_melted_df.drop(concat_melted_df.index[del_index_list], inplace = True)
    concat_melted_df = concat_melted_df.reset_index(drop = True)

    ### menu가 아예 없었던 date 지워주기
    no_date_index = list(np.where(concat_melted_df.groupby(['date']).count()[concat_melted_df.groupby(['date']).count() == 0].notnull().values == True)[0])
    no_date_list = list(concat_melted_df.groupby(['date']).count().iloc(axis = 0)[no_date_index].index.values)
    no_menu_index = [index for index, val in enumerate(concat_melted_df['date']) if val in no_date_list]
    concat_melted_df.drop(no_menu_index, axis = 0, inplace = True)

    ### menu는 있었던 date지만 특정 끼니에 음식이 없었던 경우, empty로 채워주기
    concat_melted_df = concat_melted_df.fillna('empty')
    # 참고 ! 이상하게 앞에 공백이 있는 음식이 있었음. 그런 음식들 전부 공백 제거
    concat_melted_df['menu'] = concat_melted_df['menu'].str.replace(" ", "")

    ## (3-3) 음식들의 포지션 별 (시간대 + 클래스 정보) 등장빈도 행렬구축
    # groupby로 묶어 count를 셈
    menu_by_position_count = concat_melted_df.groupby(['menu', 'position'])['date'].count().reset_index(name = 'count') 
    # 행이 menu, 열이 position인 행렬로 cast하기
    menu_by_position_label_mat = menu_by_position_count.pivot(index = 'menu', columns = 'position', values = 'count') 
    menu_by_position_label_mat = menu_by_position_label_mat.fillna(0)

    # 식단데이터에 존재하는 음식들
    foods_in_menu = menu_by_position_label_mat.index.tolist()

    # 음식데이터에 존재하는 음식들
    foods_all = food_df['name'].tolist()

    # 음식데이터에 존재하는 음식들 중 식단데이터에 존재 않는 음식들
    foods_not_in_menu = [val for i, val in enumerate(foods_all) if val not in foods_in_menu]

    # 식단데이터에 존재하는 음식들 중 음식데이터에 존재 않는 음식들
    foods_not_in_ingreds = [val for i, val in enumerate(foods_in_menu) if val not in foods_all] # 이 친구들은 menu_by_position_mat에서 제거해야함

    # foods_not_in_menu들의 원한 인코딩 행렬 만들어주기 (귤은 밥포지션에 나올 수 없다는 식의 prior에 기반한 것)
    non_menu_by_position_label_mat = pd.DataFrame(np.zeros((len(foods_not_in_menu), 14)), index = foods_not_in_menu)

    for index, val in enumerate(foods_not_in_menu):
        if food_df[food_df['name'] == val]['group'].tolist()[0] == 0:
            non_menu_by_position_label_mat.iloc[index, 0] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 1:
            target_position = [2, 9]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 2:
            target_position = [3, 10]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 3:
            target_position = [4, 5, 11, 12]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 4:
            target_position = [6, 13]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 5:
            target_position = [0, 7]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        elif food_df[food_df['name'] == val]['group'].tolist()[0] == 6:
            target_position = [1, 8]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1
        else:
            target_position = [2, 9]
            non_menu_by_position_label_mat.iloc[index, target_position] = 1


    ## 포지션 보상을 위해 One-Hot 인코딩 행렬 만들어주기
    # data + prior 기반한 모든 음식들의 원핫 인코딩 행렬
    menu_by_position_label_mat = pd.concat([menu_by_position_label_mat, non_menu_by_position_label_mat], axis = 0) 
    menu_by_position_label_mat = menu_by_position_label_mat.drop(foods_not_in_ingreds)
    menu_by_position_label_mat = menu_by_position_label_mat.reindex(food_df['name'])
    menu_by_position_label_ones = copy.deepcopy(menu_by_position_label_mat) 
    menu_by_position_label_ones[menu_by_position_label_ones > 0] = 1
    menu_by_position_label_ones = menu_by_position_label_ones

    # transition_matrix = menu_by_position_label_mat.transpose()
    # transition_= menu_by_position_label_ones.transpose()

    return(
        concat_melted_df, 
        menu_by_position_count, menu_by_position_label_mat, menu_by_position_label_ones
        )


#%%
#%%
def get_initial_sample(target_foods_df, number_of_slots):

    # (menu_by_position_label_ones 랑 mat을 class에선 전역변수로 해줘야함)
    # (또한, 저 menu_by_label_ones와 mat을 food_df만 넣어서 생성해주는 함수를 만들필요 있음)
    ## !!!! 중요, 그냥 food_df, final_sampled_food_df 등을 넣으면 one_hot_encoding을 만드는 함수 필요
    ## !!!! 혹은, 한번 만들어 놓고 전역변수로 지정하여 그냥 불러오기로 사용하는게 나을지도.

    one_hot_encoding = copy.deepcopy(menu_by_position_label_ones)
    distribution = copy.deepcopy(menu_by_position_label_mat)

    for index, menu in enumerate(one_hot_encoding.index.tolist()):
        selected_position = one_hot_encoding.columns.tolist()
        one_hot = pd.DataFrame.sample(one_hot_encoding.iloc[index], n = 1, weights = distribution.iloc[index]).index[0]
        selected_position.pop(one_hot)
        one_hot_encoding.iloc[index, selected_position] = 0

    sampled_food_list = rd.sample(target_foods_df['name'].tolist(), number_of_slots)
    
    print('초기 샘플링 음식 리스트')
    print(sampled_food_list)

    return(sampled_food_list, one_hot_encoding)
 

# %%
## 상태 반환 함수
def get_state(sampled_food_list, one_hot_encoding):
    one_hot_encoding_sample = one_hot_encoding.loc[sampled_food_list, :]
    np.set_printoptions(suppress = True)

    # sampled_food_list의 index가져오기
    food_indices = []
    for ind, val in enumerate(sampled_food_list):
        if val in list(food_df['name']):
            food_indices.append(food_df[food_df['name'] == val].index[0])

    # (1) 영양소
    nutrition_state = food_df.iloc[food_indices, 1:22].sum(axis = 0).drop(['weight'])

    # (3) 포지션
    for ind, val in enumerate(one_hot_encoding_sample.index):
        val_group = food_df[food_df['name'] == val]['group']

        col = one_hot_encoding_sample.iloc[ind, :][one_hot_encoding_sample.iloc[ind, :] > 0].index[0]
        one_hot_encoding_sample.iloc[ind, col] = int(val_group)

    position_state = one_hot_encoding_sample.sum(axis = 0)

    # if 'empty' in one_hot_encoding_sample.index: # (식단에 empty가 등장했다면)
    #     one_hot_encoding_sample = one_hot_encoding_sample.drop(['empty'])
    #     position_state = one_hot_encoding_sample.sum(axis = 0)
    # else: # (식단에 empty가 등장하지 않았다면)
    #     position_state = one_hot_encoding_sample.sum(axis = 0)

    # # empty인 포지션엔 사실 아무것도 없는것이므로 1을 0으로 대체해야함
    # empty_index = [i for i, val in enumerate(sampled_food_list) if val == 'empty'] 
    # position_state[empty_index] = 0

    print("position_state is :", position_state.tolist())
    # print("position_sum is :", sum(position_state.tolist()))

    # (4) 상태 벡터 만들기
    state = np.concatenate((nutrition_state, position_state), axis = None)

    # (5) 반환값
    return(state)


# %%
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
# %%
# def Thompson(action_size, reward_depth, Thompson_Tensor, reward, action):
#     x = linspace(.01, .99, 99) # 베타분포의 실수공간 범위
#     Action_Prob_Dist = np.ones([action_size, x.shape[0]])

#     sampled_list = []

#     for i in range(action_size):
#         alpha_val = int(Thompson_Tensor[i, 0, reward]) # 모든 행위 i의 성공 횟수 alpha
#         beta_val = int(Thompson_Tensor[i, 1, reward]) # 모든 행위 i의 실패 횟수 beta

#         # print(alpha_val)
#         # print(beta_val)
#         # 각 행위의 베타 분포 action_beta 계산 
#         action_beta = (1 / beta(alpha_val, beta_val)) * x ** (alpha_val - 1) * (1 - x) ** (beta_val - 1)
#         Action_Prob_Dist[i,] = action_beta

#         # action_beta에서 아무 값 샘플링
#         sampled_action = np.random.choice(action_beta, size = 1) 

#         # 샘플링된 action_beta 값의 index
#         candidate_action = np.where(action_beta == sampled_action)[0] 
#         selected_action = int(np.random.choice(candidate_action, size = 1))
#         # # 샘플링된 action_beta가 1개보다 많다면
#         # if len(candidate_action) > 1: 
#         #    selected_action = int(np.random.choice(candidate_action, size = 1)) # 어느 하나의 index를 선택하여 int값 반환
        
#         # # 샘플링된 action_beta가 단일하다면
#         # else: 
#         #     selected_action = int(candidate_action) # 그 샘플의 int값 반환

#         # 샘플링된 action_beta의 x 구간에서의 값 반환
#         sampled_x = x[selected_action]

#         # 각 action 별로 샘플링된 action_beta의 x 구간에서의 값을 list에 담기
#         sampled_list.append(sampled_x)

#     # Q-learning으로 선택된 행위의 상대적 선호도 (선택 확률) 계산
#     Prefer_Score = exp(sampled_list[action]) / exp(sum(sampled_list))
#     print('Prefer_Score :', Prefer_Score)

#     return(Prefer_Score, Thompson_Tensor, Action_Prob_Dist)


def Beta_Score(action_size, reward_depth, Thompson_Tensor, reward, action):
    x = linspace(.01, .99, 99) # 베타분포의 실수공간 범위
    Action_Prob_Dist = np.ones([action_size, x.shape[0]])

    sampled_list = []

    for i in range(action_size):
        alpha_val = int(Thompson_Tensor[i, 0, reward]) # 모든 행위 i의 성공 횟수 alpha
        beta_val = int(Thompson_Tensor[i, 1, reward]) # 모든 행위 i의 실패 횟수 beta

        # print(alpha_val)
        # print(beta_val)
        # 각 행위의 베타 분포 action_beta 계산 
        action_beta = (1 / beta(alpha_val, beta_val)) * x ** (alpha_val - 1) * (1 - x) ** (beta_val - 1)
        Action_Prob_Dist[i,] = action_beta

        # action_beta에서 아무 값 샘플링
        sampled_action = np.random.choice(action_beta, size = 1) 

        # 샘플링된 action_beta 값의 index
        candidate_action = np.where(action_beta == sampled_action)[0]
        if len(candidate_action) == 0:
            print('action_beta :', action_beta)
            print('sampled_action :', sampled_action)
        selected_action = int(np.random.choice(candidate_action, size = 1)) # 어느 하나의 index를 선택하여 int값 반환

        # # 샘플링된 action_beta가 1개보다 많다면
        # if len(candidate_action) > 1: 
        #    selected_action = int(np.random.choice(candidate_action, size = 1)) # 어느 하나의 index를 선택하여 int값 반환
        
        # # 샘플링된 action_beta가 단일하다면
        # else: 
        #     selected_action = int(candidate_action) # 그 샘플의 int값 반환

        # 샘플링된 action_beta의 x 구간에서의 값 반환
        sampled_x = x[selected_action]

        # 각 action 별로 샘플링된 action_beta의 x 구간에서의 값을 list에 담기
        sampled_list.append(sampled_x)

    # Q-learning으로 선택된 행위의 상대적 선호도 (선택 확률) 계산
    Prefer_Score = exp(sampled_list[action]) / exp(sum(sampled_list))
    # Prefer_Score = sampled_list[action]
    print('Prefer_Score :', Prefer_Score)

    # Thompson Sampling으로 action 선택
    # Thompson_action = np.argmax(sampled_list)
    # Thompson_Score = sampled_list[Thompson_action] / sum(sampled_list)
    # print('Thompson_Score :', Thompson_Score)

    # if Prefer_Score == Thompson_Score:
    #     Prefer_Score = 1

    return(Prefer_Score, Thompson_Tensor, Action_Prob_Dist)

def Beta_Update(decision, action, Thompson_Tensor):

    action_vector = np.array(range(action_size))
    Non_action = np.delete(action_vector, action)

    # 여기서 max_Reward는 아직 갱신되기 전 max_Reward임. 
    # 즉, 현재 reward로 max_reward가 갱신되어야 하나 갱신하지 않은 상태

    # Reward > max_Reward라면
    if decision == 0:
        # 선택된 행위의 성공 횟수 + 1
        Thompson_Tensor[action, 0, pre_reward] += 1

        # 선택된 행위 외 행위들의 실패 횟수 + 1
        Thompson_Tensor[Non_action, 1, pre_reward] += 1

    # Reward < max_Reward라면
    # else:
    #     # 모든 행위들의 실패 횟수 + 1 해줌
    #     Thompson_Tensor[action_vector, 1, pre_reward] += 1

    return(Thompson_Tensor)




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
# 선택된 행위에 따라 움직이는 함수: 즉 step함수임
def step_update(target_foods_df, sampled_food_list, action, state_vector, target_reward, activated_reward_mat): 
    check_list = dict()

    # 최대보상의 초기회상
    if target_reward < 0: # pre_reward가 음수면, reward (3) - pre_reward (-1) = 4와 같이, 실제 보상이상을 얻었다고 인식되므로
        target_reward = 0 # pre_reward가 음수일 경우 0으로 세팅.

    # actions는 self.action로 전역변수로 받아야함

    # position 기준 샘플링 음식 선택 
    # (menu_by_position_label_ones 랑 mat을 class에선 전역변수로 해줘야함)
    # (또한, 저 menu_by_label_ones와 mat을 targets_food_df만 넣어서 생성해주는 함수를 만들필요 있음)

    distribution = menu_by_position_label_mat.loc[:, action]
    selected_food_sample = pd.DataFrame.sample(distribution, n = 1, weights = distribution)
    new_food = selected_food_sample.index.tolist()[0]

    # new_food = np.random.choice(np.array(target_foods_df['name']), size = 1).tolist()[0]
    old_food = sampled_food_list[action]
    sampled_food_list[action] = new_food
    # print("new_food added Sample :", sampled_food_list)

    one_hot_encoding_update = copy.deepcopy(one_hot_encoding) 
    one_hot_encoding_update.loc[new_food, :] = 0
    one_hot_encoding_update.loc[new_food, action] = 1

    next_state_vector = get_state(sampled_food_list, one_hot_encoding_update)
    reward, activated_reward = check_reward(next_state_vector, sampled_food_list)
    
    activated_reward_mat[reward, :] += activated_reward

    reward_gradient = (reward - target_reward)

    # if reward_gradient > 0:
    #     reward = reward
    # elif reward_gradient == 0:
    #     reward = 0
    #     # reward = reward
    # else:
    #     reward = reward_gradient 
    

    # 만약 reward가 pre_reward보다 크다면 해당 action에서 해당 new_food가 나올 빈도(확률)를 +1 해주기
    # if reward >= target_reward:
    if reward_gradient >= 0:
        action_reward_distribution.loc[action, reward] += 1
        action_MAX_reward_distribution.loc[action, max_reward] += 1
        # menu_by_position_label_mat.loc[new_food, action] += 1

    # 반환값
    # print('state_vector updated to : ', next_state_vector)
    print('reward obtained : ', reward)

    check_list['state'] = next_state_vector
    check_list['state_food_list'] = sampled_food_list
    check_list['reward'] = reward
    check_list['reward_gradient'] = reward_gradient

    # action_dist = action_distribution
    # food_dist = food_distribution
    
    return(check_list, one_hot_encoding_update, activated_reward_mat)

# %%
# 수평보상을 계산하는 함수
def check_reward(state_vector, food_list):
    activated_reward = np.repeat(0, reward_depth)

    # 변수 선언
    Pos2_group = food_df['group'][food_df['name'] == food_list[2]].tolist()[0] # 밥
    Pos3_group = food_df['group'][food_df['name'] == food_list[3]].tolist()[0] # 국
    Pos9_group = food_df['group'][food_df['name'] == food_list[9]].tolist()[0] # 밥
    Pos10_group = food_df['group'][food_df['name'] == food_list[10]].tolist()[0] # 국

    Pos6_group = food_df['group'][food_df['name'] == food_list[6]].tolist()[0] # 김치
    Pos13_group = food_df['group'][food_df['name'] == food_list[13]].tolist()[0] # 김치

    Pos0_group = food_df['group'][food_df['name'] == food_list[0]].tolist()[0] # 오전간식의 간식 
    Pos1_group = food_df['group'][food_df['name'] == food_list[1]].tolist()[0] # 오전간식의 유제품 
    Pos7_group = food_df['group'][food_df['name'] == food_list[7]].tolist()[0] # 오후간식의 간식 
    Pos8_group = food_df['group'][food_df['name'] == food_list[8]].tolist()[0] # 오후간식의 유제품 

    Pos4_group = food_df['group'][food_df['name'] == food_list[4]].tolist()[0] # 오전반찬의 주찬 
    Pos5_group = food_df['group'][food_df['name'] == food_list[5]].tolist()[0] # 오전반찬의 부찬 
    Pos11_group = food_df['group'][food_df['name'] == food_list[11]].tolist()[0] # 오후반찬의 주찬 
    Pos12_group = food_df['group'][food_df['name'] == food_list[12]].tolist()[0] # 오후반찬의 부찬 

    Pos0_LiqSol = food_df['Liq_Sol'][food_df['name'] == food_list[0]].tolist()[0] # 오전간식의 간식 
    Pos1_LiqSol = food_df['Liq_Sol'][food_df['name'] == food_list[1]].tolist()[0] # 오전간식의 유제품 
    Pos7_LiqSol = food_df['Liq_Sol'][food_df['name'] == food_list[7]].tolist()[0] # 오후간식의 간식 
    Pos8_LiqSol = food_df['Liq_Sol'][food_df['name'] == food_list[8]].tolist()[0] # 오후간식의 유제품 

    target_index = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    food_list = np.array(food_list)
    ing_vector = []
    for i, val in enumerate(list(food_list[[target_index]])):
        ing_vector.append(int(food_df[food_df['name'] == val]['ing_group'].values[0]))


    cookLV_vector = []
    for i, val in enumerate(list(food_list[[target_index]])):
        cookLV_vector.append(int(food_df[food_df['name'] == val]['cooking_level'].values[0]))


    # # 오전간식, 점심, 오후간식, 저녁 칼로리 비율 제약
    # snack1 = [] # 오전간식
    # snack2 = [] # 오후간식
    # main1 = []  # 점심
    # main2 = []  # 저녁

    # for ind, val in enumerate(sampled_food_list) :
    #     if ind in [0, 1]:
    #         snack1 += food_df['weight'][food_df['name'] == val].tolist()
    #     if ind in [2, 3, 4, 5, 6]:
    #         main1 += food_df['weight'][food_df['name'] == val].tolist()
    #     if ind in [7, 8]:
    #         snack2 += food_df['weight'][food_df['name'] == val].tolist()
    #     if ind in [9, 10, 11, 12, 13]:
    #         main2 += food_df['weight'][food_df['name'] == val].tolist()

    nutrition_vec = state_vector[0:20] ## 영양소 벡터
    position_vec = state_vector[20:34]  ## 포지션 벡터
    rewards = 0 ## 보상 초기화
    # print('initial_Reward :', rewards)

    error = 0.10 ## 오차허용범위 10%
    target_kcal =  1400 - 280  ## 권장 칼로리
    gen_kcal = nutrition_vec[0] ## 생성 칼로리

    ## 생성 탄수화물 칼로리 산출
    carbonhydrate_kcal = 4 * nutrition_vec[1] 

    ## 생성 단백질 칼로리 산출
    protein_kcal = 4 * nutrition_vec[5] 

    ## 생성 지방 칼로리 산출
    fat_kcal = 9 * nutrition_vec[2] 

    ## 총 생성 칼로리 대비 생성 탄수화물 칼로리 비율 산출
    carbonhydrate_ratio = (carbonhydrate_kcal / gen_kcal) * 100 

    ## 총 생성 칼로리 대비 생성 단백질 칼로리 비율 산출
    protein_ratio = (protein_kcal / gen_kcal) * 100 

    ## 총 생성 칼로리 대비 생성 지방 칼로리 비율 산출
    fat_ratio = (fat_kcal / gen_kcal) * 100 

    ## 총 생성 칼로디 대비 생성 탄단지 칼로리합 비율 산출
    total_ratio = (carbonhydrate_kcal + fat_kcal + protein_kcal) * 100 / gen_kcal

    ## 포화지방산 및 트랜스지방산 적정중량 산출
    target_sturated_fat = (gen_kcal * 0.08) / 9
    target_trans_fat = (gen_kcal * 0.01) / 9
    
    # 포지션에 속한 음식들의 총 개수
    pos_total_food_amount = position_vec.sum() 

    # 음식의 개수가 1을 초과한 포지션의 개수
    # pos_each_food_amount = sum(list(position_vec > 1)) 

    # if (
    #     (pos_each_food_amount <= 1)  
    #     # (sum(position_vec[2:7]) >= 4 and sum(position_vec[2:7]) <= 5) and 
    #     # (sum(position_vec[9:14]) >= 4 and sum(position_vec[9:14]) <= 5)
    #     ):
    #     rewards +=1

        # 보상기준 1: 영양소
        ### 참고 1: 탄-지-단의 비율합은 100%가 되도록 해야함.
        ### 참고 2: 탄수화물과 단백질은 1g당 4kcal, 지방은 1g당 9kcal임.

    # if (
    #     # 죽류 : group = 0
    #     # 밥류 : group = 1
    #     # 일품 : group = 2
    #     # 국류 : group = 3
    #     # 주찬 : group = 4
    #     # 부찬 : group = 5
    #     # 김치류 : group = 6
    #     # 간식 : group = 7
    #     # 유제품 : group = 8
    #     # Empty : group = 9
    #     # Pos2_group / Pos9_group : 밥 or 일품 or empty 클래스(1 or 2 or 9)
    #     # Pos3_group / Pos10_group : 국 or 일품 or empty 클래스(3 or 2 or 9)
    #     ## 보상기준 3-8: 간식류 및 유제품류 음식이 적절하게 배치되었을 때 보상
    #     (Pos0_group in [0, 7]) or # 포지션 0은 죽(0) 아니면 간식(7)
    #     (Pos1_group == 8) or # 포지션 1은 유제품(8)
    #     (Pos2_group in [1, 2]) or # 포지션 2는 밥(1) 혹은 일품 (2)
    #     (Pos3_group in [3, 9]) or # 포지션 3은 국(3) 혹은 empth(9)
    #     (Pos4_group == 4) or # 포지션 4 및 포지션 11은 주찬(4)
    #     (Pos11_group == 4) or
    #     (Pos5_group == 5) or # 포지션 5 및 포지션 12는 부찬(5)
    #     (Pos12_group == 5) or
    #     (Pos7_group == 7) or # 포지션 7은 간식(7)
    #     (Pos8_group == 8) or # 포지션 8은 유제품(8)
    #     (Pos6_group == 6) or # 포지션 6 및 포지션 13은 김치(6)
    #     (Pos13_group == 6)
    #     ):
    #     rewards += 1 # or때문에 총 12점임.
    
    # 죽류 : group = 0
    # 밥류 : group = 1
    # 일품 : group = 2
    # 국류 : group = 3
    # 주찬 : group = 4
    # 부찬 : group = 5
    # 김치류 : group = 6
    # 간식 : group = 7
    # 유제품 : group = 8

    if len(set(ing_vector)) == 10:
        rewards += 1
        activated_reward[0] = 1
    else:
        print('Duplicate Ingredient')

    if len(np.where(np.array(cookLV_vector) == 0)[0]) == 1:
        rewards += 1
        activated_reward[1] += 1
    else:
        print('Cook_Level')

    # 중복음식이 없다면 보상 +1
    if len(set(food_list)) == 14:
        rewards += 1
        activated_reward[2] = 1
    else:
        print('reward 0')

    # if Pos0_group in [0.0, 7.0]:  # 포지션 0은 죽(0) 아니면 간식(7)
    #     rewards += 1
    # else:
    #     print('reward 1')

    # if Pos1_group == 8.0:  # 포지션 1은 유제품(8)
    #     rewards += 1
    # else:
    #     print('reward 2')

    # if Pos2_group in [1.0, 2.0, 9.0]: # 포지션 2는 밥(1) 혹은 일품(2) 혹은 empty(9)
    #     rewards += 1
    # else:
    #     print('reward 3')

    # if Pos3_group in [2.0, 3.0, 9.0]: # 포지션 3은 일품(2) 혹은 국(3) 혹은 empty(9)
    #     rewards += 1
    # else:
    #     print('reward 4')

    # if Pos3_group != 2.0: # 포지션 3은 일품(2)이어선 안됨
    #     rewards += 1
    # else:
    #     print('reward 5')

    # if Pos9_group in [1.0, 2.0, 9.0]: # 포지션 9는 밥(1) 혹은 일품(2) 혹은 empty(9)
    #     rewards += 1
    # else:
    #     print('reward 6')

    # if Pos10_group in [2.0, 3.0, 9.0]: # 포지션 10은 일품(2) 혹은 국(3) 혹은 empty(9)
    #     rewards += 1
    # else:
    #     print('reward 7')

    # if Pos10_group != 2.0: # 포지션 10은 일품(2)이어선 안됨
    #     rewards += 1
    # else:
    #     print('reward 8')

    # if Pos4_group == 4.0: # 포지션 4 및 포지션 11은 주찬(4)
    #     rewards += 1
    # else:
    #     print('reward 9')

    # if Pos11_group == 4.0: # 포지션 4 및 포지션 11은 주찬(4)
    #     rewards += 1
    # else:
    #     print('reward 10')

    # if Pos5_group == 5.0: # 포지션 5 및 포지션 12는 부찬(5)
    #     rewards += 1
    # else:
    #     print('reward 11')

    # if Pos12_group == 5.0: # 포지션 5 및 포지션 12는 부찬(5)
    #     rewards += 1
    # else:
    #     print('reward 12')

    # if Pos7_group == 7.0: # 포지션 7은 간식(7)
    #     rewards += 1
    # else:
    #     print('reward 13')

    # if Pos8_group == 8.0: # 포지션 8은 유제품(8)
    #     rewards += 1
    # else:
    #     print('reward 14')

    # if Pos6_group == 6.0: # 포지션 6 및 포지션 13은 김치(6)
    #     rewards += 1
    # else:
    #     print('reward 15')

    # if Pos13_group == 6.0: # 포지션 6 및 포지션 13은 김치(6)
    #     rewards += 1
    # else:
    #     print('reward 16')

    #총 17점임.

    # Pos2_group / Pos9_group : 밥 or 일품 or empty 클래스(1 or 2 or 9)
    # Pos3_group / Pos10_group : 국 or 일품 or empty 클래스(3 or 2 or 9)
    # Pos2_group + Pos3_group 합이 4 (밥 + 국)와 11 (일품 + Empty)만 됨.
    # Pos9_group + Pos10_group 합이 4 (밥 + 국)와 11 (일품 + Empty)만 됨.

    Lunch_Base = Pos2_group + Pos3_group
    if Lunch_Base in [4.0, 11.0] and Pos2_group != Pos3_group:
        rewards += 1
        activated_reward[3] = 1
    else:
        print('reward 17')

    Dinner_Base = Pos9_group + Pos10_group
    if Dinner_Base in [4.0, 11.0] and Pos9_group != Pos10_group:
        rewards += 1  
        activated_reward[4] = 1
    else:
        print('reward 18')


    # if(
    #     ## 보상기준 3-8: 간식 - 간식의 조합이 적절할 때 보상
    #     # 미상 + 액체류 : 0 + 1 = 1
    #     # 미상 + 고체류 : 0 + 2 = 2
    #     # 미상 + 죽류 : 0 + 5 = 5
    #     # 액체류 + 액체류 : 1 + 1 = 2
    #     # 액체류 + 고체류 : 1 + 2 = 3 (o)
    #     # 고체류 + 고체류 : 2 + 2 = 4 
    #     # 액체류 + 죽류 : 1 + 5 = 6 (o)
    #     # 고체류 + 죽류 : 2 + 5 = 7 (o)
    #     # 3, 5, 6 중에서 나와야함

    #     # 오전간식은 반드시 액체유(1)와 고체류(2) 중 하나 + 죽류로 구성되어야 함
    #     (Pos0_LiqSol + Pos1_LiqSol >= 3) and 

    #     # 오전간식에 죽류 + 죽류, 미상 + 죽류로 구성되면 안됨
    #     (Pos0_LiqSol + Pos1_LiqSol != 4 and Pos0_LiqSol + Pos1_LiqSol != 10 and Pos0_LiqSol + Pos1_LiqSol != 5) or

    #     # 오후간식은 반드시 액체유(1)와 고체류(2) 중 하나 + 죽류로 구성되어야 함         
    #     (Pos7_LiqSol + Pos8_LiqSol >= 3) and 

    #     # 오전간식에 죽류 + 죽류, 미상 + 죽류로 구성되면 안됨
    #     (Pos7_LiqSol + Pos8_LiqSol != 4 and Pos7_LiqSol + Pos8_LiqSol != 10 and Pos7_LiqSol + Pos8_LiqSol != 5)   
    #     ):
    #     rewards +=1 # or 때문에 총 2점임


    # 오전간식은 반드시 액체류(1)와 고체류(2) 중 하나 + 죽류(5)로 구성되어야 함
    # 오전간식에 죽류 + 죽류, 미상 + 죽류로 구성되면 안됨
    Morning_Dessert = Pos0_LiqSol + Pos1_LiqSol
    if Morning_Dessert in [3.0]:
        rewards += 1
        activated_reward[5] = 1
    else:
        print('reward 19')

        ## 보상기준 3-8: 간식 - 간식의 조합이 적절할 때 보상
        # 미상 + 액체류 : 0 + 1 = 1
        # 미상 + 고체류 : 0 + 2 = 2
        # 미상 + 죽류 : 0 + 5 = 5
        # 액체류 + 액체류 : 1 + 1 = 2
        # 액체류 + 고체류 : 1 + 2 = 3 (o)
        # 고체류 + 고체류 : 2 + 2 = 4 
        # 액체류 + 죽류 : 1 + 5 = 6 (o)
        # 고체류 + 죽류 : 2 + 5 = 7 (o)


    # 오후간식은 반드시 액체류(1)와 고체류(2) 중 하나로 구성되어야 함             
    Evening_Dessert = Pos7_LiqSol + Pos8_LiqSol
    if Evening_Dessert in [3.0]:
        rewards += 1 
        activated_reward[6] = 1
    else:
        print('reward 20')


    # if (
    #     # Pos2_group / Pos9_group : 밥 or 일품 or empty 클래스(1 or 2 or 9)
    #     # Pos3_group / Pos10_group : 국 or 일품 or empty 클래스(3 or 2 or 9)
    #     # Pos2_group + Pos3_group 합이 4 (밥 + 국)와 11 (일품 + Empty)만 됨.
    #     # Pos9_group + Pos10_group 합이 4 (밥 + 국)와 11 (일품 + Empty)만 됨.
    #     ((Pos2_group + Pos3_group) in [4, 11] and Pos3_group != 2) or
    #     ((Pos9_group + Pos10_group) in [4, 11] and Pos10_group != 2)
    #     ):
    #     rewards +=1 # 2점임
    

    # (우선순위 1: 열량 기준)
    ## 보상기준 1-1: 생성 칼로리가 권장 칼로리 (1120kcal)의 += 10% (1008 ~ 1232 kcal) 범위안에 포함될 시.
    if (
        (gen_kcal >= target_kcal * (1 - error)) and (gen_kcal <= target_kcal * (1 + error))
        ):
        rewards +=1
        activated_reward[7] = 1
    else:
        print('reward 21')

    # if ( # 하루 권장 칼로리의 8% ~ 11%를 오전간식으로.
    #     (sum(snack1) >= (0.9 * 1400 * (1 - (error)))) and (sum(snack1) <= (0.10 * 1400 * (1 + (error))))
    #     # (sum(snack2) >= (0.10 * 1400 * (1 - (error)))) and (sum(snack2) <= (0.10 * 1400 * (1 + (error)))) and
    #     # (sum(main1) >= (0.30 * 1400 * (1 - (error)))) and (sum(main1) <= (0.25 * 1400 * (1 + (error)))) and
    #     # (sum(main2) >= (0.30 * 1400 * (1 - (error)))) and (sum(main2) <= (0.25 * 1400 * (1 + (error))))
    #     ):
    #     rewards +=1
    # else:
    #     print('Snack1 Calory')

    #     # ## 보상기준 2-2-1: 개별 - 각 포지션에 속한 음식의 개수가 1 초과시 "마이너스" 보상
    #     # ## 보상기준 2-2-2: 개별 - 각 포지션에 속한 음식의 개수가 0 또는 1 일시 보상 
    #     # ## 보상기준 2-2-3: 개별 - 점심과 저녁 포지션에 속한 음식의 개수의 합이 4이상 5이하 


    # (우선순위 2: 탄단지 종합 비율 기준)
    ## 탄-단-지 총 kcal가 전체 kcal의 90% ~ 100%에 육박해야 함
    if (total_ratio >= 90) and (total_ratio <= 100):
        rewards +=1
        activated_reward[8] = 1
    else:
        print('reward 22')


    # (우선순위 3: 탄단지 개별 비율 및 중량)
    if (
        ## 보상기준 1-2: 탄수화물 - 55 ~ 70(%) 일 경우 보상
        ((carbonhydrate_ratio >= 55) and (carbonhydrate_ratio <= 70)) and 
        ## 보상기준 1-3: 지방 - 15 ~ 30(%) 일 경우 보상 
        ((fat_ratio >= 15) and (fat_ratio <= 30)) and
        ## 보상기준 1-4: 단백질 - 7 ~ 20 (%) + 최소 20g 일 때 보상
        ((protein_ratio >= 7) and (protein_ratio <= 20) and (nutrition_vec[5] >= 20))
        ):
        rewards +=1
        activated_reward[9] = 1
    else:
        print('reward 23')


    # (우선순위 2-3: 포화 및 트랜스지방산 비율)
    ## 보상기준 1-5: 포화지방산 - 8(%) 넘을 경우 "마이너스" 보상
    ## 보상기준 1-6: 트랜스지방산 - 1 (%) 넘을 경우 "마이너스" 보상
    if (
        (nutrition_vec[3] <= target_sturated_fat) and (nutrition_vec[4] <= target_trans_fat)
        ):
        rewards +=1
        activated_reward[10] = 1
    else:
        print('reward 24')


    # (우선순위 4: 그외)
    if (
        ## 보상기준 1-7: 식이섬유 - 10g ~ 30g 사이일 때 보상
        ((nutrition_vec[6] >= 10) and (nutrition_vec[6] <= 30)) and
        ## 보상기준 1-8: 칼슘 - 400mg ~ 2500mg 사이일 때 보상
        ((nutrition_vec[7] >= 400) and (nutrition_vec[7] <= 2500)) and
        ## 보상기준 1-9: 철 - 6mg ~ 40mg 사이일 때 보상
        ((nutrition_vec[8] >= 6) and (nutrition_vec[8] <= 40)) and
        ## 보상기준 1-10: 나트륨 - 500mg ~ 1000mg 사이일 때 보상
        # ((nutrition_vec[9] >= 500) and (nutrition_vec[9] <= 1000)) and
        ## 보상기준 1-11: 인 - 300mg ~ 3000mg 사이일 때 보상
        ((nutrition_vec[10] >= 300) and (nutrition_vec[10] <= 3000)) and
        ## 보상기준 1-12: 비타민 A - 230ug RE ~ 700ug RE 사이일 때 보상
        ((nutrition_vec[11] >= 230) and (nutrition_vec[11] <= 700)) and
        # ## 보상기준 1-13: 비타민 B1(Thiamine) - 0.45mg ~ 0.55mg 사이일 때 보상
        # ((nutrition_vec[12] >= 0.45) and (nutrition_vec[12] <= 0.55)) or
        # ## 보상기준 1-14: 비타민 B2(Riboflavin) - 0.54mg ~ 0.66mg 사이일 때 보상
        # ((nutrition_vec[13] >= 0.54) and (nutrition_vec[13] <= 0.66)) or
        # ## 보상기준 1-15: 비타민 B6 - 0.9mg ~ 35mg 사이일 때 보상
        # ((nutrition_vec[14] >= 0.9) and (nutrition_vec[14] <= 35)) or
        # ## 보상기준 1-16: 비타민 B12(Cyanocobalamin) -  0.9mg ~ 1.1mg 사이일 때 보상
        # ((nutrition_vec[15] >= 0.9) and (nutrition_vec[15] <= 1.1)) or
        ## 보상기준 1-17: 비타민 C(Ascorbic Acid) - 30mg ~ 500mg 사이일 때 보상
        ((nutrition_vec[16] >= 30) and (nutrition_vec[16] <= 500)) and
        ## 보상기준 1-18: 비타민 D - 2ug ~ 35ug 사이일 때 보상
        ((nutrition_vec[17] >= 2) and (nutrition_vec[17] <= 35)) 
        # ## 보상기준 1-20: 아연(Zinc) - 3.6mg ~ 9mg 사이일 때 보상
        # ((nutrition_vec[19] >= 3.6) and (nutrition_vec[19] <= 9)) or
        # ## 보상기준 1-19: 비타민 E - 250mg a-TE 넘을 경우 "마이너스" 보상
        # (nutrition_vec[18] <= 250) 
        ):
        rewards +=1
        activated_reward[11] = 1
    else:
        print('reward 25')

    return (rewards, activated_reward)


#%%
#%%
def train_model(state, action, next_action, reward, reward_grad, reward_mean, next_state, epsilon, epsilon_decay, epsilon_min, discount_factor, model):
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
# %%
def parameter_read(param_set):

    file_path = Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Parameter_new.csv')
    if file_path.is_file() == False:
        with open(file_path, 'w', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = list(param_set.keys()))
            writer.writeheader()

    f = open(file_path, mode = 'a', encoding = 'utf-8', newline = '')
    writer = csv.DictWriter(f, fieldnames = list(param_set.keys()))
    writer.writerow(param_set)
    f.close()
# %%

# %%

# %%
# (1) 데이터 구축
## food 데이터 읽어와 food 데이터 프레임 생성
food_df = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/foods.csv', encoding = 'CP949')
## 20번째 위치에 group변수가 있으니, 인덱스 위치는 16
food_df.columns.get_loc('group')

## empty 벡터 추가
food_df.loc(axis = 0)[1725] = ['empty'] + [0] * (len(food_df.columns) - 1)
food_df.loc(axis = 1)['group'][1725] = 9
food_df.loc(axis = 1)['ing_group'][1725] = 154

## menu 데이터 시간대별로 나눠서 데이터 읽어온 후 데이터 프레임 생성
menu_morning_df = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_morning.csv', encoding = 'CP949')
menu_lunch_df = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_lunch.csv', encoding = 'CP949')
menu_afternoon_df = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_afternoon.csv', encoding = 'CP949')
menu_dinner_df = pd.read_csv('/home/messy92/Leo/Project_Gosin/Data/real_menu_dinner.csv', encoding = 'CP949')
list_of_menus_df = [menu_morning_df, menu_lunch_df, menu_afternoon_df, menu_dinner_df]

## (2) 데이터 전처리
concat_melted_df, menu_by_position_count, menu_by_position_label_mat, menu_by_position_label_ones = data_preprocessing(food_df, list_of_menus_df)

## (3) 초기값은 균일확률로 처리
menu_by_position_label_mat[menu_by_position_label_mat >= 0] = 0

## (4) 확정적 슬롯으로 처리
jook = food_df['name'][food_df['group'] == 0].tolist()
bab = food_df['name'][food_df['group'] == 1].tolist()
ilpum = food_df['name'][food_df['group'] == 2].tolist()
gook = food_df['name'][food_df['group'] == 3].tolist()
juchan = food_df['name'][food_df['group'] == 4].tolist()
buchan = food_df['name'][food_df['group'] == 5].tolist()
kimchi = food_df['name'][food_df['group'] == 6].tolist()
gansik = food_df['name'][food_df['group'] == 7].tolist()
euje = food_df['name'][food_df['group'] == 8].tolist()
empty_food = food_df['name'][food_df['group'] == 9].tolist()

# 오전 간식
action0_condition = jook + gansik + euje + empty_food
action1_condition = gansik + euje + empty_food
# 오후 간식
action7_condition = gansik + euje + empty_food
action8_condition = gansik + euje + empty_food
# 밥
action2_condition = bab + ilpum + empty_food
action9_condition = bab + ilpum + empty_food
# 국
action3_condition = gook + empty_food
action10_condition = gook + empty_food
# 주찬
action4_condition = juchan + empty_food
action11_condition = juchan + empty_food
# 부찬
action5_condition = buchan + empty_food
action12_condition = buchan + empty_food
# 김치
action6_condition = kimchi + empty_food
action13_condition = kimchi + empty_food

condition0_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action0_condition]
condition1_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action1_condition]
condition2_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action2_condition]
condition3_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action3_condition]
condition4_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action4_condition]
condition5_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action5_condition]
condition6_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action6_condition]
condition7_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action7_condition]
condition8_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action8_condition]
condition9_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action9_condition]
condition10_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action10_condition]
condition11_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action11_condition]
condition12_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action12_condition]
condition13_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action13_condition]

menu_by_position_label_mat.iloc[condition0_index, 0] = 1
menu_by_position_label_mat.iloc[condition1_index, 1] = 1
menu_by_position_label_mat.iloc[condition2_index, 2] = 1
menu_by_position_label_mat.iloc[condition3_index, 3] = 1
menu_by_position_label_mat.iloc[condition4_index, 4] = 1
menu_by_position_label_mat.iloc[condition5_index, 5] = 1
menu_by_position_label_mat.iloc[condition6_index, 6] = 1
menu_by_position_label_mat.iloc[condition7_index, 7] = 1
menu_by_position_label_mat.iloc[condition8_index, 8] = 1
menu_by_position_label_mat.iloc[condition9_index, 9] = 1
menu_by_position_label_mat.iloc[condition10_index, 10] = 1
menu_by_position_label_mat.iloc[condition11_index, 11] = 1
menu_by_position_label_mat.iloc[condition12_index, 12] = 1
menu_by_position_label_mat.iloc[condition13_index, 13] = 1

menu_by_position_label_ones = menu_by_position_label_mat


## (4) 행위벡터 정의
## 길이 113의 행위 (전략 or 정책) 벡터 생성 (14 (포지션) * 9 (8 클래스 + 1 empty클래스) + 1 (변함없거나))
num_of_position = 14
actions = num_of_position # 선택 포지션
actions = list(range(actions))

## (5) 전이행렬 정의 (균일확률분포)
transition_matrix = np.ones([len(menu_by_position_label_mat.index), len(menu_by_position_label_mat.index)])
transition_matrix = pd.DataFrame(transition_matrix, index = menu_by_position_label_mat.index.tolist(), columns = menu_by_position_label_mat.index.tolist()).astype(int)

## (6) 행위 / 음식 분포 (업데이트)
action_distribution = np.ones([len(actions)]).astype(int).tolist()
food_distribution = np.ones([food_df.shape[0]]).astype(int).tolist()

# %%

# %%
# for 문 도는 구간
num_of_samples = 2000
# state_size = 14
state_size = 20
# state_size = 34
action_size = len(actions)
reward_depth = 12

model = build_model(state_size, action_size)

action_reward_distribution = np.zeros([action_size, reward_depth + 1])
action_reward_distribution = pd.DataFrame(action_reward_distribution.astype(int))
col = [x for x in list(range(reward_depth + 1))] # 각 reward를 얻기 이전에 행한 행동 분포 (0~8)
action_reward_distribution.columns = col

action_MAX_reward_distribution = np.zeros([action_size, reward_depth + 1])
action_MAX_reward_distribution = pd.DataFrame(action_MAX_reward_distribution.astype(int))
col = [x for x in list(range(reward_depth  + 1))] # 각 reward_depth에서 행하는 행동 분포 (0~8)
action_MAX_reward_distribution.columns = col

## (10) 엡실론 초기화 (초기화 하면 안도니ㅏ...?)
epsilon = 0.2
# epsilon_decay = 0.99999
epsilon_decay = 1
epsilon_min = 0.01

Mean_Episodic_rewards = []
Mean_Episodic_score = []
Mean_epsilon_trajectory = []
completed_menu = []
# MarkovChain_list = []
data_save = dict()

# time_step = []

activated_reward_mat = np.zeros([reward_depth + 1, reward_depth])

for k in range(num_of_samples):
    # num_decision = 2 # max_reward 돌파 or not
    # Thompson_Tensor = np.ones([action_size, num_decision, reward_depth + 1]) # 매 샘플마다 갱신

    # 모든 초기화 변수들 입력해주기
    ## (5) 초기식단 샘플 확보
    if k == 0:
        sampled_food_list, one_hot_encoding = get_initial_sample(food_df, 14)
        fixed_sampled_food_list = copy.deepcopy(sampled_food_list)
        fixed_one_hot_encoding = copy.deepcopy(one_hot_encoding)
    else:
        sampled_food_list = copy.deepcopy(fixed_sampled_food_list)
        one_hot_encoding = copy.deepcopy(fixed_one_hot_encoding)

    ## (7) score & discount_factor 초기화
    score = 0
    discount_factor = 0.99

    ## (8) 초기 상태벡터 받아오기
    state_vector = get_state(sampled_food_list, one_hot_encoding)
    state_vector

    ## (11) 정지조건
    Done = False

    i = 0
    num_updates = 0
    Episodic_rewards = []
    Episodic_score = []
    # Episodic_loss = []
    epsilon_trajectory = []
    max_reward = 0
    num_rewarded = 0
    pre_reward = 0
    reward = 0
    reward_mean = 0

    w_mse_vector = []
    mse_loss_vector = []

    w_origin = model.layers[2].get_weights()[1]

    while not Done: # Done이 False인 한
        i += 1
        one_hot_encoding_origin = copy.deepcopy(one_hot_encoding)

        ## (10) actions = {move, to_where, actions_to_action_space} 만들어, 현재 행위 (move & to_where) 예측
        ## action_space, actions, actions_to_action_space, model, epsilon 전부 _init_에 선언될 전역 변수
        action, state_vector = get_action(state_vector, actions, model, epsilon)

        print('Action Selected', action)

        # old_food 저장 & pre_reward 저장
        old_food = sampled_food_list[action]  
        pre_reward = copy.deepcopy(reward)
        # target_reward = copy.deepcopy(pre_reward)
        target_reward = copy.deepcopy(max_reward)

        ## (11) 현재 행위로부터 다음상태 (next_state) 예측
        check_list, one_hot_encoding, activated_reward_mat = step_update(food_df, sampled_food_list, action, state_vector, target_reward, activated_reward_mat)

        ## (12) 상태, 인덱스, 리스트, 보상 업데이트
        next_state_vector = check_list['state']
        reward = check_list['reward']
        reward_mean = (reward_mean + reward) / i

        reward_grad = check_list['reward_gradient']

        ## (13) 다음 행위 (next_move & next_to_where) 예측
        ## action_space, actions, actions_to_action_space, model, epsilon 전부 _init_에 선언될 전역 변수
        next_action, next_state_vector2 = get_action(next_state_vector, actions, model, epsilon)

        # if(check_list['Depth_IN'] == "Yes"):
        num_updates += 1
        
        # # Beta분포에 기반한 확률점수 계산
        # Prefer_Score, Thompson_Tensor, Action_Prob_Dist = Beta_Score(action_size, reward_depth, Thompson_Tensor, reward, action)

            # elif k_old < k:
            #     plt.figure(figsize=(12, 8))
            #     figure, axes = plt.subplots(nrows = 2, ncols = 1)
                
            #     axes[0].plot(range(len(w_mse_vector)), w_mse_vector, 'b-')
            #     axes[1].plot(range(len(mse_loss_vector)), mse_loss_vector, 'r-')
            #     axes[0].set_title('Last Layer Weight MSE')
            #     axes[1].set_title('Loss')
            #     axes[0].grid()
            #     axes[1].grid()
            #     figure.tight_layout(rect=[0, 0.0001, 1, 0.90])
            #     figure.suptitle(str(k) + '-th trial ' + str(i) + '-th iter', fontsize = 20)
            #     figure.subplots_adjust(top = 0.85)
            #     plt.savefig(Path('Generated_Sample/Last_Layer_Weight_MSE.png'))

        ## (14) 모델 학습
        epsilon, mse_loss = train_model(state_vector, action, next_action, reward, reward_grad, reward_mean, next_state_vector, epsilon, epsilon_decay, epsilon_min, discount_factor, model)
        w_update = model.layers[2].get_weights()[1] # 3번쨰 layer의 weights 받기
        w_error = (w_origin - w_update)
        w_mse = np.matmul(w_error, w_error.T)
        w_mse_vector.append(w_mse)
        mse_loss_vector.append(mse_loss)
        print('w_mse is : ', w_mse)
        w_origin = w_update

        # ## 베타분포 파라미터 업데이트
        # if reward > pre_reward:
        #     decision = 0
        # else:
        #     decision = 1
        # Thompson_Tensor = Beta_Update(decision, action, Thompson_Tensor)

        # Back Sampling (최고난이도 보상은 딱 한번 확보되기 떄문에, 중간 단계 보상도 한번만 확보되게 하기 위해서)
        if reward < max_reward:
            sampled_food_list[action] = old_food
            # next_state_vector = get_state(sampled_food_list, one_hot_encoding_origin)
            one_hot_encoding = copy.deepcopy(one_hot_encoding_origin)

        # MarkovChain_list.append(sampled_food_list[action])
        score += check_list['reward']
        if check_list['reward'] > max_reward:
            max_reward = copy.deepcopy(check_list['reward'])
    
        if check_list['reward'] > 0:
            num_rewarded += 1  

        # 갱신
        state_vector = next_state_vector
        print("Max_Reward", max_reward)
        print('final_sample:', sampled_food_list)

        print(
        "Sample :", k, "-th sample" ,
        "Episodes : ", i,
        "Updates : ", num_updates,
        "Score :", score,
        "Max_Reward", max_reward,
        "num_rewarded", num_rewarded,
        "fixed_Sample", fixed_sampled_food_list
        )

        # Episodic_loss.append(mse_loss)
        Episodic_rewards.append(reward)
        Episodic_score.append(score)
        epsilon_trajectory.append(epsilon)

        if max_reward == reward_depth:
            Done = True
            print("Episode Ends! Congratulation!")

        # mse 그려주기
        # time_step.append(copy.deepcopy(i))
        if i != 0 and i % 300 == 0 or Done == True:
            # if k_old == k:
            # if i > 300 and i % 300 == 0 : # 저장 mse의 길이가 900을 넘으면
                # del w_mse_vector[:300] # 제일 앞의 300개 mse 삭제
                # del mse_loss_vector[:300] # 제일 앞의 300개 mse 삭제
            plt.figure(figsize=(12, 8))
            figure, axes = plt.subplots(nrows = 2, ncols = 1)
            
            axes[0].plot(range(len(w_mse_vector)), w_mse_vector, 'b-')
            axes[1].plot(range(len(mse_loss_vector)), mse_loss_vector, 'r-')
            axes[0].set_title('Last Layer Weight MSE')
            axes[1].set_title('Loss')
            axes[0].grid()
            axes[1].grid()
            figure.tight_layout(rect=[0, 0.0001, 1, 0.90])
            figure.suptitle(str(k) + '-th trial ' + str(i) + '-th iter', fontsize = 20)
            figure.subplots_adjust(top = 0.85)
            plt.savefig(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Last_Layer_Weight_MSE.png'))


    if ((k != 0) and (k % 10 == 0)):
        # action_reward_distribution.plot(kind = 'bar')
        action_reward_distribution.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/action_reward.csv'))
        action_MAX_reward_distribution.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Action_Per_Reward_Depth.csv'))
        # MarkovChain_list.to_csv(Path('Generated_Sample/MarkovChain.csv'))
        Action_Reward_PATH = pd.DataFrame(np.matrix(action_reward_distribution) + np.matrix(action_MAX_reward_distribution)).T
        Action_Reward_PATH.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Action_Reward_PATH.csv'))
        # pd.DataFrame(Thompson_Tensor).to_csv(Path('Generated_Sample/Thompson_RewardAction_Tuple.csv'))
        # menu_by_position_label_mat.to_csv(Path('Generated_Sample/Menu_By_Position.csv'))
        # plt.show()
        # top10_list = transition_matrix.sum(axis = 1).sort_values(ascending = False)[0:10].index.tolist()
        # transition_matrix.loc[top10_list, top10_list].style.background_gradient(axis = 1, cmap = 'Blues')
    if (k == (num_of_samples - 1)):
        The_Message = pd.Series(str('EveryThing Is Done ~!'))
        The_Message.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Finish_Letter.csv'))

    Mean_rewards = sum(Episodic_rewards) / len(Episodic_rewards)
    Mean_epsilon = sum(epsilon_trajectory) / len(epsilon_trajectory)
    Mean_Episodic_rewards.append(Mean_rewards)
    Mean_epsilon_trajectory.append(Mean_epsilon)
    completed_menu.append(check_list['state_food_list'])
    
    data_save['Num_Sample'] = k
    data_save['Len_Episode'] = i
    data_save['Num_Update'] = num_updates
    data_save['Total_Score'] = score
    data_save['Mean_Rewards'] = Mean_rewards
    data_save['Mean_Epsilon'] = Mean_epsilon
    gmName_list = []
    for i in range(0, len(check_list['state_food_list'])) : 
        data_save['Generated_Menu_{}'.format(i)] = check_list['state_food_list'][i]
    for j in range(0, len(check_list['state'][0:20])):
        data_save['Nutrition_{}'.format(j)] = check_list['state'][0:20][j]
    parameter_read(data_save)


plt.subplot(211)
plt.plot(list(range(k + 1)), Mean_Episodic_rewards, 'rs--')
plt.xlabel('Sample')
plt.ylabel('Mean Reward')
plt.subplot(212)
plt.plot(list(range(k + 1)), Mean_epsilon_trajectory, 'bo--')
plt.xlabel('Episode')
plt.ylabel('Sample')
plt.suptitle('Empirical Result')
plt.show()



 # random_state 고정할껀지 안할껀지 결정해야함...
 # sarsa의 로칼 옵티마 문제점
 # 정지조건... 평균 실제 식단과 비슷하면 ㅈ어지???
 # 그냥 탐색을 final_candidate_food_df가 아니라 food_df에서 하는것도 방법...
 # 기준완화... +-10% -> +-15%??
 # 그냥 가장 이상적인 state_vector와 sample된 state_vector의 코사인 유사도, L2norm 등으로 보상을 줄까??
 # decay만큼 epsilon 감소하는거는 reward가 양수일때만 하는것도 고려하기
 # empty같은 음식을 적극적으로 replace sampling할 수있는 방법은..?


 # 재료기반 샘플스페이스확정은 포기해야할듯... 샘플링 풀이 너무 작아짐.
 # 한 에피소드도 너무 시간이 너무 오래걸림...
 # 그래도 칼로리를 학습한 경향은 확실히 확인하였음.
 # 내 생각엔 그냥 영양소, 클래스, 포지션 정보를 한꺼번에 진짜 데이터와 GAN을 통해 비교하여 통으로
 # 얼마나 생성 식단벡터가 실제 식단 벡터와 유사한지를 reward로 주는게 가장 깔끔 명료할듯...이런방식은 학습시간이 너무 오래걸리고
 # 보상간 층위로 인해 특정 보상위주로 학습되고 하층 보상은 그 효과가 뭉개짐.
 # 하층 보상뿐만아니라 보상의 달성 확률이 낮은 보상도 그 효과가 무시됨,
# %%
# csv파일로 생성
# with open("tmp.csv", "w", newline = "") as f:
#     writer = csv.writer(f)
#     writer.writerows(action_reward_distribution)

# %%
# pd.Series(np.diag(transition_matrix), index = [transition_matrix.index, transition_matrix.columns]).sort_values(ascending = False)

# action_reward 분포 (action 기준 색칠:column-wise)
action_reward_distribution.style.background_gradient(cmap = 'Blues', axis = 1)

# food_distribution 상위 100개 음식들의 소속 group
food_df.iloc[pd.Series(food_distribution).sort_values(ascending = False)[0:100].index.tolist(), :]['group'].value_counts().sort_index().plot(kind = 'bar')

# action_distribution 분포
pd.Series(action_distribution).plot(kind = 'bar')

# action이 reward를 얻기 위해 행하는 path
Action_Reward_PATH = pd.read_csv('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Generated_Sample/Action_Reward_PATH.csv', index_col = 0)
Action_Reward_PATH.style.background_gradient(cmap = 'Blues', axis = 1)
Action_Reward_PATH_Long = copy.deepcopy(Action_Reward_PATH)
Action_Reward_PATH_Long['index'] = Action_Reward_PATH_Long.index
Action_Reward_PATH_melt = pd.melt(Action_Reward_PATH_Long, id_vars = ['index'], value_vars = Action_Reward_PATH_Long.columns.values.tolist()[:14])

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(Action_Reward_PATH_melt['index'], Action_Reward_PATH_melt['variable'], Action_Reward_PATH_melt['value'], cmap = cm.Blues)
plt.show()
# pd.read_csv(file_path, encoding = 'CP949')

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
