# %%
import pandas as pd
import numpy as np
import collections as col
import random as rd
import csv
from pathlib import Path
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from numpy import linspace, exp
from scipy.special import beta, gamma

import utils
from utils import *
import Agent
from Agent import *

import Environment
from Environment import *

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

## (4) 확정적 슬롯으로 처리
jook = food_df['name'][food_df['group'] == 0].tolist()          # 죽 그룹에 속한 음식들 리스트
bab = food_df['name'][food_df['group'] == 1].tolist()           # 밥 그룹에 속한 음식들 리스트
ilpum = food_df['name'][food_df['group'] == 2].tolist()         # 일품 그룹에 속한 음식들 리스트
gook = food_df['name'][food_df['group'] == 3].tolist()          # 국 그룹에 속한 음식들 리스트
juchan = food_df['name'][food_df['group'] == 4].tolist()        # 주찬 그룹에 속한 음식들 리스트
buchan = food_df['name'][food_df['group'] == 5].tolist()        # 부찬 그룹에 속한 음식들 리스트
kimchi = food_df['name'][food_df['group'] == 6].tolist()        # 김치 그룹에 속한 음식들 리스트
gansik = food_df['name'][food_df['group'] == 7].tolist()        # 간식 그룹에 속한 음식들 리스트
euje = food_df['name'][food_df['group'] == 8].tolist()          # 유제품 그룹에 속한 음식들 리스트
empty_food = food_df['name'][food_df['group'] == 9].tolist()    # 빈칸 그룹에 속한 음식들 리스트

## 각 행위 (슬롯)에 해당하는 음식들의 조건 (condition) 설정
### : 예를 들어, 행위 0에 해당되는 음식들은 죽, 간식, 유제품, 빈칸이라는 그룹에 속한 음식들이어야 한다는 조건을 설정

# 오전 간식
action0_condition = jook + gansik + euje + empty_food    # 오전간식 클래스: 행위 (슬롯) 0번에 해당되는 음식들 리스트
action1_condition = gansik + euje + empty_food           # 오전간식 클래스: 행위 (슬롯) 1번에 해당되는 음식들 리스트
# 오후 간식
action7_condition = gansik + euje + empty_food           # 오후간식 클래스: 행위 (슬롯) 7번에 해당되는 음식들 리스트
action8_condition = gansik + euje + empty_food           # 오후간식 클래스: 행위 (슬롯) 8번에 해당되는 음식들 리스트
# 밥
action2_condition = bab + ilpum + empty_food             # 밥 클래스: 행위 (슬롯) 2번에 해당되는 음식들 리스트
action9_condition = bab + ilpum + empty_food             # 밥 클래스: 행위 (슬롯) 9번에 해당되는 음식들 리스트
# 국
action3_condition = gook + empty_food                    # 국 클래스: 행위 (슬롯) 3번에 해당되는 음식들 리스트
action10_condition = gook + empty_food                   # 밥 클래스: 행위 (슬롯) 10번에 해당되는 음식들 리스트
# 주찬
action4_condition = juchan + empty_food                  # 주찬 클래스: 행위 (슬롯) 4번에 해당되는 음식들 리스트
action11_condition = juchan + empty_food                 # 주찬 클래스: 행위 (슬롯) 11번에 해당되는 음식들 리스트
# 부찬
action5_condition = buchan + empty_food                  # 부찬 클래스: 행위 (슬롯) 5번에 해당되는 음식들 리스트   
action12_condition = buchan + empty_food                 # 부찬 클래스: 행위 (슬롯) 12번에 해당되는 음식들 리스트
# 김치
action6_condition = kimchi + empty_food                  # 김치 클래스: 행위 (슬롯) 6번에 해당되는 음식들 리스트
action13_condition = kimchi + empty_food                 # 김치 클래스: 행위 (슬롯) 6번에 해당되는 음식들 리스트

### 각 슬롯 조건에 부함하는 음식들의 인덱스 리스트 저장
condition0_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action0_condition]     # 행위 (슬롯) 0번에 해당되는 음식들의 인덱스
condition1_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action1_condition]     # 행위 (슬롯) 1번에 해당되는 음식들의 인덱스
condition2_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action2_condition]     # 행위 (슬롯) 2번에 해당되는 음식들의 인덱스
condition3_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action3_condition]     # 행위 (슬롯) 3번에 해당되는 음식들의 인덱스
condition4_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action4_condition]     # 행위 (슬롯) 4번에 해당되는 음식들의 인덱스
condition5_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action5_condition]     # 행위 (슬롯) 5번에 해당되는 음식들의 인덱스
condition6_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action6_condition]     # 행위 (슬롯) 6번에 해당되는 음식들의 인덱스
condition7_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action7_condition]     # 행위 (슬롯) 7번에 해당되는 음식들의 인덱스
condition8_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action8_condition]     # 행위 (슬롯) 8번에 해당되는 음식들의 인덱스
condition9_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action9_condition]     # 행위 (슬롯) 9번에 해당되는 음식들의 인덱스
condition10_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action10_condition]   # 행위 (슬롯) 10번에 해당되는 음식들의 인덱스
condition11_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action11_condition]   # 행위 (슬롯) 11번에 해당되는 음식들의 인덱스
condition12_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action12_condition]   # 행위 (슬롯) 12번에 해당되는 음식들의 인덱스
condition13_index = [food_index for food_index, val in enumerate(menu_by_position_label_mat.index.tolist()) if val in action13_condition]   # 행위 (슬롯) 13번에 해당되는 음식들의 인덱스

# 위의 condition을 따르는 음식 - 슬롯넘버 형태의 원핫인코딩 행렬 만들어주기
## menu_by_position_label_mat : foods x position 행렬; index (row)는 음식명, column은 슬롯넘버인 행렬; 데이터 기반 분포; 
## menu_by_position_label_ones : foods x position 행렬; index (row)는 음식명, column은 슬롯넘버인 행렬; 행위 (슬롯)에 해당하는 조건 기반 균일분포;
menu_by_position_label_ones = copy.deepcopy(menu_by_position_label_mat)
menu_by_position_label_ones[menu_by_position_label_ones >= 0] = 0

menu_by_position_label_ones.iloc[condition0_index, 0] = 1
menu_by_position_label_ones.iloc[condition1_index, 1] = 1 
menu_by_position_label_ones.iloc[condition2_index, 2] = 1
menu_by_position_label_ones.iloc[condition3_index, 3] = 1
menu_by_position_label_ones.iloc[condition4_index, 4] = 1
menu_by_position_label_ones.iloc[condition5_index, 5] = 1
menu_by_position_label_ones.iloc[condition6_index, 6] = 1
menu_by_position_label_ones.iloc[condition7_index, 7] = 1
menu_by_position_label_ones.iloc[condition8_index, 8] = 1
menu_by_position_label_ones.iloc[condition9_index, 9] = 1
menu_by_position_label_ones.iloc[condition10_index, 10] = 1
menu_by_position_label_ones.iloc[condition11_index, 11] = 1
menu_by_position_label_ones.iloc[condition12_index, 12] = 1
menu_by_position_label_ones.iloc[condition13_index, 13] = 1


## (4) 행위벡터 정의
## 길이 113의 행위 (전략 or 정책) 벡터 생성 (14 (포지션) * 9 (8 클래스 + 1 empty클래스) + 1 (변함없거나))
num_of_position = 14
actions = num_of_position # 선택 포지션
actions = list(range(actions))

# %%
# for 문 도는 구간
num_of_samples = 200
# state_size = 14
state_size = 20
# state_size = 34
action_size = len(actions)
reward_depth = 11

model = build_model(state_size, action_size)
Env = env(food_df, menu_by_position_label_mat, menu_by_position_label_ones, action_size, reward_depth)

## (10) 엡실론 초기화 (초기화 하면 안도니ㅏ...?)
epsilon = 0.0
# epsilon_decay = 0.99999
epsilon_decay = 1
epsilon_min = 0.01

Mean_Episodic_rewards = []
Mean_Episodic_score = []
Mean_epsilon_trajectory = []
completed_menu = []
data_save = dict()

activated_reward_mat = np.zeros([reward_depth + 1, reward_depth])

for k in range(num_of_samples):

    # 모든 초기화 변수들 입력해주기
    ## (5) 초기식단 샘플 확보
    if k == 0:
        sampled_food_list, one_hot_encoding = Env.get_initial_sample(food_df, 14)
        fixed_sampled_food_list = copy.deepcopy(sampled_food_list)
        fixed_one_hot_encoding = copy.deepcopy(one_hot_encoding)
    else:
        sampled_food_list = copy.deepcopy(fixed_sampled_food_list)
        one_hot_encoding = copy.deepcopy(fixed_one_hot_encoding)

    ## (7) score & discount_factor 초기화
    score = 0
    discount_factor = 0.99

    ## (8) 초기 상태벡터 받아오기
    state_vector = Env.get_state(sampled_food_list, one_hot_encoding)
    state_vector

    ## (11) 정지조건
    Done = False

    i = 0
    num_updates = 0
    Episodic_rewards = []
    Episodic_score = []
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
        action, state_vector = get_action(state_vector, actions, model, epsilon)  # action_space, actions, actions_to_action_space, model, epsilon 전부 _init_에 선언될 전역 변수


        print('Action Selected', action)

        # old_food 저장 & pre_reward 저장
        old_food = sampled_food_list[action]  
        pre_reward = copy.deepcopy(reward)
        # target_reward = copy.deepcopy(pre_reward)
        target_reward = copy.deepcopy(max_reward)

        ## (11) 현재 행위로부터 다음상태 (next_state) 예측
        check_list, one_hot_encoding, activated_reward_mat, action_reward_distribution, action_MAX_reward_distribution = Env.step_update(food_df, sampled_food_list, action, state_vector, target_reward, activated_reward_mat, max_reward)

        ## (12) 상태, 인덱스, 리스트, 보상 업데이트
        next_state_vector = check_list['state']
        reward = check_list['reward']
        reward_mean = (reward_mean + reward) / i

        reward_grad = check_list['reward_gradient']

        ## (13) 다음 행위 (next_move & next_to_where) 예측
        next_action, next_state_vector2 = get_action(next_state_vector, actions, model, epsilon) # action_space, actions, actions_to_action_space, model, epsilon 전부 _init_에 선언될 전역 변수


        num_updates += 1 # update 개수 세기

        ## (14) 모델 학습
        epsilon, mse_loss = train_model(state_vector, action, next_action, reward, reward_grad, reward_mean, next_state_vector, epsilon, epsilon_decay, epsilon_min, discount_factor, model, reward_depth, max_reward)
        w_update = model.layers[2].get_weights()[1] # 3번쨰 layer의 weights 받기
        w_error = (w_origin - w_update)
        w_mse = np.matmul(w_error, w_error.T)
        w_mse_vector.append(w_mse)
        mse_loss_vector.append(mse_loss)
        print('w_mse is : ', w_mse)
        w_origin = w_update


        # Back Sampling
        if reward < max_reward:                                       # 새로 확보한 보상이 이전까지 확보된 최대 보상보다 작다면 
            sampled_food_list[action] = old_food                      # 다시 이전 food_list로 리턴
            one_hot_encoding = copy.deepcopy(one_hot_encoding_origin) # one_hot_encoding도 이전으로 리턴

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
            plt.savefig(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Deep Q-Learning/gen_sample/Last_Layer_Weight_MSE.png'))


    if ((k != 0) and (k % 10 == 0)):
        action_reward_distribution.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Deep Q-Learning/gen_sample/action_reward.csv'))
        action_MAX_reward_distribution.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Deep Q-Learning/gen_sample/Action_Per_Reward_Depth.csv'))
        Action_Reward_PATH = pd.DataFrame(np.matrix(action_reward_distribution) + np.matrix(action_MAX_reward_distribution)).T
        Action_Reward_PATH.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Deep Q-Learning/gen_sample/Action_Reward_PATH.csv'))
    if (k == (num_of_samples - 1)):
        The_Message = pd.Series(str('EveryThing Is Done ~!'))
        The_Message.to_csv(Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/Deep Q-Learning/gen_sample/Finish_Letter.csv'))

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
# pd.Series(np.diag(transition_matrix), index = [transition_matrix.index, transition_matrix.columns]).sort_values(ascending = False)

# action_reward 분포 (action 기준 색칠:column-wise)
action_reward_distribution.style.background_gradient(cmap = 'Blues', axis = 1)

# food_distribution 상위 100개 음식들의 소속 group
food_df.iloc[pd.Series(food_distribution).sort_values(ascending = False)[0:100].index.tolist(), :]['group'].value_counts().sort_index().plot(kind = 'bar')

# action_distribution 분포
pd.Series(action_distribution).plot(kind = 'bar')

# action이 reward를 얻기 위해 행하는 path
Action_Reward_PATH = pd.read_csv('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/gen_sample/Action_Reward_PATH.csv', index_col = 0)
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
