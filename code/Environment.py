#%%
import pandas as pd
import numpy as np
import collections as col
import random as rd
import copy

class env:
    def __init__(self, food_df, menu_by_position_label_mat, menu_by_position_label_ones, action_size, reward_depth):
        self.food_df = food_df
        # self.menu_by_position_label_mat = menu_by_position_label_mat
        self.menu_by_position_label_mat = menu_by_position_label_ones
        self.menu_by_position_label_ones = menu_by_position_label_ones

        self.one_hot_encoding = None
        
        self.action_size = action_size
        self.reward_depth = reward_depth

        self.action_reward_distribution = np.zeros([self.action_size, self.reward_depth + 1])
        self.action_reward_distribution = pd.DataFrame(self.action_reward_distribution.astype(int))
        col = [x for x in list(range(self.reward_depth + 1))] # 각 reward를 얻기 이전에 행한 행동 분포 (0~8)
        self.action_reward_distribution.columns = col

        self.action_MAX_reward_distribution = np.zeros([self.action_size, self.reward_depth + 1])
        self.action_MAX_reward_distribution = pd.DataFrame(self.action_MAX_reward_distribution.astype(int))
        col = [x for x in list(range(self.reward_depth  + 1))] # 각 reward_depth에서 행하는 행동 분포 (0~8)
        self.action_MAX_reward_distribution.columns = col


    ## 최초의 식단 (음식리스트)과 식단의 원핫인코딩 행렬 구성
    def get_initial_sample(self, target_foods_df, number_of_slots):

        # (menu_by_position_label_ones 랑 mat을 class에선 전역변수로 해줘야함)
        # (또한, 저 menu_by_label_ones와 mat을 food_df만 넣어서 생성해주는 함수를 만들필요 있음)
        ## !!!! 중요, 그냥 food_df, final_sampled_food_df 등을 넣으면 one_hot_encoding을 만드는 함수 필요
        ## !!!! 혹은, 한번 만들어 놓고 전역변수로 지정하여 그냥 불러오기로 사용하는게 나을지도.

        self.one_hot_encoding = copy.deepcopy(self.menu_by_position_label_ones)
        distribution = copy.deepcopy(self.menu_by_position_label_mat)

        for index, menu in enumerate(self.one_hot_encoding.index.tolist()):
            selected_position = self.one_hot_encoding.columns.tolist()
            one_hot = pd.DataFrame.sample(self.one_hot_encoding.iloc[index, :], n = 1, weights = distribution.iloc[index, :]).index[0]
            selected_position.pop(one_hot)
            self.one_hot_encoding.iloc[index, selected_position] = 0

        sampled_food_list = rd.sample(target_foods_df['name'].tolist(), number_of_slots)
        
        print('초기 샘플링 음식 리스트')
        print(sampled_food_list)

        return(sampled_food_list, self.one_hot_encoding)
    
    ## 상태 반환 함수
    def get_state(self, sampled_food_list, one_hot_encoding):
        self.one_hot_encoding = one_hot_encoding
        one_hot_encoding_sample = self.one_hot_encoding.loc[sampled_food_list, :]
        np.set_printoptions(suppress = True)

        # sampled_food_list의 index가져오기
        food_indices = []
        for ind, val in enumerate(sampled_food_list):
            if val in list(self.food_df['name']):
                food_indices.append(self.food_df[self.food_df['name'] == val].index[0])

        # (1) 영양소
        nutrition_state = self.food_df.iloc[food_indices, 1:22].sum(axis = 0).drop(['weight'])

        # (3) 포지션
        for ind, val in enumerate(one_hot_encoding_sample.index):
            val_group = self.food_df[self.food_df['name'] == val]['group']
            col = one_hot_encoding_sample.iloc[ind, :][one_hot_encoding_sample.iloc[ind, :] > 0].index[0]
            one_hot_encoding_sample.iloc[ind, col] = int(val_group)

        position_state = one_hot_encoding_sample.sum(axis = 0)

        print("position_state is :", position_state.tolist())
        # print("position_sum is :", sum(position_state.tolist()))

        # (4) 상태 벡터 만들기
        state = np.concatenate((nutrition_state, position_state), axis = None)

        # (5) 반환값
        return(state)

    # 선택된 행위에 따라 움직이는 함수: 즉 step함수임
    def step_update(self, target_foods_df, sampled_food_list, action, state_vector, target_reward, activated_reward_mat, max_reward): 
        check_list = dict()

        if target_reward < 0: # pre_reward가 음수면, reward (3) - pre_reward (-1) = 4와 같이, 실제 보상이상을 얻었다고 인식되므로
            target_reward = 0 # pre_reward가 음수일 경우 0으로 세팅.

        # actions는 self.action로 전역변수로 받아야함

        # position 기준 샘플링 음식 선택 
        # (menu_by_position_label_ones 랑 mat을 class에선 전역변수로 해줘야함)
        # (또한, 저 menu_by_label_ones와 mat을 targets_self.food_df만 넣어서 생성해주는 함수를 만들필요 있음)

        distribution = self.menu_by_position_label_mat.loc[:, action]
        selected_food_sample = pd.DataFrame.sample(distribution, n = 1, weights = distribution)
        new_food = selected_food_sample.index.tolist()[0]

        # new_food = np.random.choice(np.array(target_foods_df['name']), size = 1).tolist()[0]
        old_food = sampled_food_list[action]
        sampled_food_list[action] = new_food
        # print("new_food added Sample :", sampled_food_list)

        one_hot_encoding_update = copy.deepcopy(self.one_hot_encoding) 
        one_hot_encoding_update.loc[new_food, :] = 0
        one_hot_encoding_update.loc[new_food, action] = 1

        next_state_vector = self.get_state(sampled_food_list, one_hot_encoding_update)
        reward, activated_reward = self.check_reward(next_state_vector, sampled_food_list)
        
        activated_reward_mat[reward, :] += activated_reward

        reward_gradient = (reward - target_reward)
        

        # 만약 reward가 pre_reward보다 크다면 해당 action에서 해당 new_food가 나올 빈도(확률)를 +1 해주기
        # if reward >= target_reward:
        if reward_gradient >= 0:
            self.action_reward_distribution.loc[action, reward] += 1
            self.action_MAX_reward_distribution.loc[action, max_reward] += 1
            # menu_by_position_label_mat.loc[new_food, action] += 1

        # 반환값
        # print('state_vector updated to : ', next_state_vector)
        print('reward obtained : ', reward)

        check_list['state'] = next_state_vector
        check_list['state_food_list'] = sampled_food_list
        check_list['reward'] = reward
        check_list['reward_gradient'] = reward_gradient

        return(check_list, one_hot_encoding_update, activated_reward_mat, self.action_reward_distribution, self.action_MAX_reward_distribution)

    # 수평보상을 계산하는 함수
    def check_reward(self, state_vector, food_list):
        activated_reward = np.repeat(0, self.reward_depth)

        # 변수 선언
        Pos2_group = self.food_df['group'][self.food_df['name'] == food_list[2]].tolist()[0] # 밥
        Pos3_group = self.food_df['group'][self.food_df['name'] == food_list[3]].tolist()[0] # 국
        Pos9_group = self.food_df['group'][self.food_df['name'] == food_list[9]].tolist()[0] # 밥
        Pos10_group = self.food_df['group'][self.food_df['name'] == food_list[10]].tolist()[0] # 국

        Pos6_group = self.food_df['group'][self.food_df['name'] == food_list[6]].tolist()[0] # 김치
        Pos13_group = self.food_df['group'][self.food_df['name'] == food_list[13]].tolist()[0] # 김치

        Pos0_group = self.food_df['group'][self.food_df['name'] == food_list[0]].tolist()[0] # 오전간식의 간식 
        Pos1_group = self.food_df['group'][self.food_df['name'] == food_list[1]].tolist()[0] # 오전간식의 유제품 
        Pos7_group = self.food_df['group'][self.food_df['name'] == food_list[7]].tolist()[0] # 오후간식의 간식 
        Pos8_group = self.food_df['group'][self.food_df['name'] == food_list[8]].tolist()[0] # 오후간식의 유제품 

        Pos4_group = self.food_df['group'][self.food_df['name'] == food_list[4]].tolist()[0] # 오전반찬의 주찬 
        Pos5_group = self.food_df['group'][self.food_df['name'] == food_list[5]].tolist()[0] # 오전반찬의 부찬 
        Pos11_group = self.food_df['group'][self.food_df['name'] == food_list[11]].tolist()[0] # 오후반찬의 주찬 
        Pos12_group = self.food_df['group'][self.food_df['name'] == food_list[12]].tolist()[0] # 오후반찬의 부찬 

        Pos0_LiqSol = self.food_df['Liq_Sol'][self.food_df['name'] == food_list[0]].tolist()[0] # 오전간식의 간식 
        Pos1_LiqSol = self.food_df['Liq_Sol'][self.food_df['name'] == food_list[1]].tolist()[0] # 오전간식의 유제품 
        Pos7_LiqSol = self.food_df['Liq_Sol'][self.food_df['name'] == food_list[7]].tolist()[0] # 오후간식의 간식 
        Pos8_LiqSol = self.food_df['Liq_Sol'][self.food_df['name'] == food_list[8]].tolist()[0] # 오후간식의 유제품 


        # 같은 재료 클러스터
        target_index = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
        food_list = np.array(food_list)
        ing_vector = []
        for i, val in enumerate(list(food_list[[target_index]])):
            ing_vector.append(int(self.food_df[self.food_df['name'] == val]['ing_group'].values[0]))


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

        # 음식 재료간 중복이 없다면 보상 +1
        if len(set(ing_vector)) == 10:
            rewards += 1
            activated_reward[0] = 1
        else:
            print('Duplicate Ingredient')


        # 중복음식이 없다면 보상 +1
        if len(set(food_list)) == 14:
            rewards += 1
            activated_reward[1] = 1
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

        # 점심 베이스 (밥, 국)의 구성 클래스가 (1) 밥 + 국 = 4.0, (2) 일품 + empty = 11.0일 경우 보상 +1
        Lunch_Base = Pos2_group + Pos3_group
        if Lunch_Base in [4.0, 11.0] and Pos2_group != Pos3_group:
            rewards += 1
            activated_reward[2] = 1
        else:
            print('reward 17')


        # 저녁 베이스 (밥, 국)의 구성 클래스가 (1) 밥 + 국 = 4.0, (2) 일품 + empty = 11.0일 경우 보상 +1
        Dinner_Base = Pos9_group + Pos10_group
        if Dinner_Base in [4.0, 11.0] and Pos9_group != Pos10_group:
            rewards += 1  
            activated_reward[3] = 1
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


        ## 보상기준 3-8: 간식 - 간식의 조합이 적절할 때 보상
        # 미상 + 액체류 : 0 + 1 = 1
        # 미상 + 고체류 : 0 + 2 = 2
        # 미상 + 죽류 : 0 + 5 = 5
        # 액체류 + 액체류 : 1 + 1 = 2
        # 액체류 + 고체류 : 1 + 2 = 3 (o)
        # 고체류 + 고체류 : 2 + 2 = 4 
        # 액체류 + 죽류 : 1 + 5 = 6 (o)
        # 고체류 + 죽류 : 2 + 5 = 7 (o)

        # 오전간식은 반드시 액체류(1)와 고체류(2) 중 하나로 구성되어야 함
        Morning_Dessert = Pos0_LiqSol + Pos1_LiqSol
        if Morning_Dessert in [3.0]:
            rewards += 1
            activated_reward[4] = 1
        else:
            print('reward 19')

        # 오후간식은 반드시 액체류(1)와 고체류(2) 중 하나로 구성되어야 함             
        Evening_Dessert = Pos7_LiqSol + Pos8_LiqSol
        if Evening_Dessert in [3.0]:
            rewards += 1 
            activated_reward[5] = 1
        else:
            print('reward 20')

        

        # (우선순위 1: 열량 기준)
        ## 보상기준 1-1: 생성 칼로리가 권장 칼로리 (1120kcal)의 += 10% (1008 ~ 1232 kcal) 범위안에 포함될 시.
        if (
            (gen_kcal >= target_kcal * (1 - error)) and (gen_kcal <= target_kcal * (1 + error))
            ):
            rewards +=1
            activated_reward[6] = 1
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
            activated_reward[7] = 1
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
            activated_reward[8] = 1
        else:
            print('reward 23')


        # (우선순위 2-3: 포화 및 트랜스지방산 비율)
        ## 보상기준 1-5: 포화지방산 - 8(%) 넘을 경우 "마이너스" 보상
        ## 보상기준 1-6: 트랜스지방산 - 1 (%) 넘을 경우 "마이너스" 보상
        if (
            (nutrition_vec[3] <= target_sturated_fat) and (nutrition_vec[4] <= target_trans_fat)
            ):
            rewards +=1
            activated_reward[9] = 1
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
            activated_reward[10] = 1
        else:
            print('reward 25')

        return (rewards, activated_reward)

# %%
