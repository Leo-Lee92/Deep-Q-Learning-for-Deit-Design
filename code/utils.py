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


def parameter_read(param_set):
    file_path = Path('/home/messy92/Leo/Project_Gosin/Q_Learning/Code/gen_sample/Parameter_new.csv')
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
