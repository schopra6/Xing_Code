import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
from collections import OrderedDict as od
from itertools import chain
import scipy
from scipy.sparse import csr_matrix
from process_xgb_data import sparcify
from statistics import median,mean,mode

user_key = 'user_id'
item_key = 'item_id'
def event_feature(df):
    columns_users = ['career_level_user','region_user','industry_id_user','discipline_id_user','country_user','experience_years_experience','edu_degree','experience_years_in_current']
   
    columns_item = ['career_level_item','region_item','industry_id_item','discipline_id_item','country_item']
    for column in columns_item:
        df['intu_'+column+'_contained'] = percentage_contained(df,'intu_'+column.replace('_item',''),column)
    for column in columns_users:
        df['intuser_'+column+'_contained'] = percentage_contained(df,'intuser_'+column.replace('_user',''),column)
    return df    
#test_interactions = pd.read_csv('../prepare_data/test_dataset/test_interaction.csv')
#Y_labels =test_interactions[['user_id' ,'item_id']].groupby('user_id')['item_id'].agg(set).reset_index().set_index('user_id')

def common_attrib(df):
    columns_users = ['jobrole_list_all']
   
    columns_item = ['tag_list_all','title_list_all']
    df_copy = df[['user_id','item_id','intuser_jobrole_list']].expand(intuser_jobrole_list)

    for column in columns_item:
        df['intu_'+column+'_contained'] = intersect(df,'intu_'+column.replace('_item',''),column)
    for column in columns_users:
        df['intuser_'+column+'_contained'] = intersect(df,'intuser_'+column.replace('_user',''),column)
    for col in something:
         df[col].apply(lambda items_list : [len(set(feature).intersection(set(featureA))) for feature in feature_list])
    return df  

def intersect(df,columnA,columnB):
   df[columnA+'_contained'] = df.apply(lambda x : len(set(x[columnA]).intersection(set(x[columnB]))) )

def clicks(df,user_key='user_id',item_key='item_id'):

    item_clicks = df.groupby(item_key)[user_key].count().reset_index(name = 'clicks')
    return item_clicks

def previous_click(positive_train_interactions,df):
    prev_click_df = positive_train_interactions[['user_id','item_id']].groupby(['user_id','item_id']).value_counts().reset_index(name='prev_count')
    df = pd.merge(df,prev_click_df, how ='left',on =['user_id','item_id'])
    df['prev_count']= df['prev_count'].fillna(0)
    return df
def Intu_Intusers(df,users,items,user_key='user_id',item_key='item_id'):  

    Intusers = df.groupby([item_key])[user_key].apply(list).reset_index()
    Intu =df.groupby([user_key])[item_key].apply(list).reset_index()
    Intucolumns =['career_level', 'discipline_id', 'industry_id','region','country','latitude','longitude','tag_list','title_list']
    Intusercolumns = [ 'career_level', 'discipline_id', 'industry_id','region','country',
       'experience_years_experience', 'experience_years_in_current','edu_degree','jobrole_list']
    for column in Intucolumns:
           Intu['intu_'+ column] = Intu[item_key].apply(lambda items_list : [items.loc[item][column] for item in items_list])
    for column in Intusercolumns:
           Intusers['intuser_'+ column] = Intusers[user_key].apply(lambda users_list : [users.loc[user][column] for user in users_list]) 
    Intu['intu_tag_list_all'] = Intu['intu_tag_list'].apply(lambda x : list(chain(*x) ))
    Intu['intu_title_list_all'] = Intu['intu_tag_list'].apply(lambda x : list(chain(*x) ))
    Intusers['intuser_jobrole_list'] = Intusers['intuser_jobrole_list'].apply(lambda x : list(chain(*x) ))
    Intu['intu_mode_career_level'] = Intu['intu_career_level'].map(lambda x : sorted(set(x), key=x.count)[-2:])
    Intu['intu_mode_industry_id'] = Intu['intu_industry_id'].map(lambda x :  sorted(set(x), key=x.count)[-2:])
    Intu['intu_mode_discipline_id'] = Intu['intu_discipline_id'].map(lambda x :  sorted(set(x), key=x.count)[-2:])  
    Intusers.drop(user_key,axis=1,inplace=True) 
    return Intu,Intusers

def jaccard_util(jaccard,test_user_ids,user_items,jaccard_threshold,item_ids,sparse_collaborative_matrix):
    for row_num,ele in enumerate(test_user_ids):
        user_indexes  =np.argwhere(jaccard[row_num]>jaccard_threshold)[:,1]
        item_indexes = set(sparse_collaborative_matrix[user_indexes,:].nonzero()[1])
        item_id_list = np.take(item_ids,list(item_indexes))
        user_items['item_id'].loc[ele] = user_items['item_id'].loc[ele]+list(item_id_list)
    return user_items
title_missed=[]
titag_missed=[]
job_role_missed=[]
combined_missed=[]    

def similarity_util(sparse_collaborative_matrix,test_user_ids,train_item_title_sparse,test_item_title_sparse,train_item_tag_sparse,test_item_tag_sparse,user_jobrole_sparse,user_items,user_to_row,item_ids,test_item_ids):
    ################################################
    title_title_similarity = intersection_similarity(train_item_title_sparse,test_item_title_sparse)
    title_tag_similarity = intersection_similarity(train_item_title_sparse,test_item_tag_sparse)
    tag_title_similarity = intersection_similarity(train_item_tag_sparse,test_item_title_sparse)
    tag_tag_similarity = intersection_similarity(train_item_tag_sparse,test_item_tag_sparse)
    jobrole_title_similarity = intersection_similarity(user_jobrole_sparse,test_item_tag_sparse)
    jobrole_tag_similarity = intersection_similarity(user_jobrole_sparse,test_item_title_sparse)
    for row_num,ele in enumerate(test_user_ids):
        testuser_i =user_to_row[ele]
        item_indexes = set(sparse_collaborative_matrix[testuser_i,:].nonzero()[1])
        titi_similar_item_indexes = list(chain(*[title_title_similarity[item_index,:].indices.tolist() for item_index in item_indexes]))
        #tita_similar_item_indexes =list(chain(*[title_tag_similarity[item_index,:].indices[np.argsort(-title_tag_similarity[item_index,:].data)][:5].tolist() for item_index in item_indexes]))
        #tati_similar_item_indexes =list(chain(*[tag_title_similarity[item_index,:].indices[np.argsort(-tag_title_similarity[item_index,:].data)][:5].tolist() for item_index in item_indexes]))
        #tata_similar_item_indexes =list(chain(*[tag_tag_similarity[item_index,:].indices[np.argsort(-tag_tag_similarity[item_index,:].data)][:3].tolist() for item_index in item_indexes]))
        #joti_similar_item_indexes =list(chain(*[jobrole_title_similarity[testuser_i,:].indices.tolist()]))
        #jota_similar_item_indexes =list(chain(*[jobrole_tag_similarity[testuser_i,:].indices[np.argsort(-jobrole_tag_similarity[testuser_i,:].data)][:5].tolist()]))
        #combined_indexes = list(set(tati_similar_item_indexes).intersection(tata_similar_item_indexes).intersection(set(jota_similar_item_indexes)))
        #title_missed.append(len(Y_labels.loc[ele].values[0] - set(np.unique(np.take(item_ids,titi_similar_item_indexes)))))
        #titag_missed.append(Y_labels.loc[ele].values[0]- set(np.unique(np.take(item_ids,tita_similar_item_indexes))))
        #job_role_missed.append(Y_labels.loc[ele].values[0] - set(np.unique(np.take(item_ids,joti_similar_item_indexes))))
        #combined_missed.append(Y_labels.loc[ele].values[0] - set(np.unique(np.take(item_ids,combined_indexes))))
        
        item_i = titi_similar_item_indexes  #tita_similar_item_indexes + combined_indexes + joti_similar_item_indexes 
        #item_i=list(chain(*item_i))
        item_id_list = np.unique(np.take(item_ids,item_i))
        item_id_list = item_id_list[np.in1d(item_id_list, test_item_ids, assume_unique=True)]
        user_items['item_id'].loc[ele]=list(set(user_items['item_id'].loc[ele]+list(item_id_list)))
    return user_items

def get_stats(list1):
    print(f' similarity mean{mean(list1)} median{median(list1)} mode {mode(list1)}')


def jaccard_similarity(matA,matB):

    intersection = matA.dot(matB.T)
    matA_sum = scipy.sum(matA,axis=1)
    matB_sum = matB.sum(axis=1)
    union = np.array(np.squeeze(matB_sum))[0] +matA_sum -intersection
    jaccard = intersection/union
    return jaccard

def click_ratio(df,user_key='user_id',item_key='item_id'):

    df = df.join(df.groupby('item_id')['created_at'].max(), on='item_id', rsuffix='_last_week_start')
    df = df.join(df.groupby('item_id')['created_at'].max(), on='item_id', rsuffix='_previous_last_week_start')
    df['created_at_last_week_start'] = df['created_at_last_week_start'] - pd.DateOffset(weeks=1)
    df['created_at_previous_last_week_start'] = df['created_at_previous_last_week_start']  - pd.DateOffset(weeks=2)

    last_week_data = df[(df['created_at'] > df['created_at_last_week_start'])]
    week_before_last_data = df[(df['created_at'] > df['created_at_previous_last_week_start']) & (df['created_at'] <= df['created_at_last_week_start'])]
    last_week_click_counts = last_week_data['item_id'].value_counts()
    week_before_last_click_counts = week_before_last_data['item_id'].value_counts()
    df =df.join(last_week_click_counts/week_before_last_click_counts, on='item_id',rsuffix='_click_ratio')
    df['item_id_click_ratio'] = df['item_id_click_ratio'].fillna(1)
    df = df[['item_id','item_id_click_ratio']].drop_duplicates(subset=['item_id'])
    return df


def user_activity(df,user_key='user_id',item_key='item_id'):

    user_events = df[[user_key,'item_id']].groupby(user_key)['item_id'].count().reset_index(name = 'user_activity')
    return user_events

def click_date_max(train):

    item_id_max = train.groupby(item_key)['created_at'].max().reset_index(name = 'item_id_max_time')
    user_id_max = train.groupby(user_key)['created_at'].max().reset_index(name = 'user_id_max_time')
    return item_id_max,user_id_max
    
def last_click_activity(df):
    import pandas as pd

# Load your interaction dataframe
    # interaction_df = pd.read_csv('interaction_data.csv')

    # Sort the dataframe by user_id and timestamp
   
    max_timestamp =  df['item_id_max_time'].max()
    df['max_click_time']=max_timestamp
    df['item_user_max_time'] = (df['item_id_max_time']-df['user_id_max_time']).dt.components['hours']
    df['item_id_max_time'] = (max_timestamp-df['item_id_max_time']).dt.components['hours']
    df['user_id_max_time'] = (max_timestamp-df['user_id_max_time']).dt.components['hours']
    return df

def content_similarity(df):

    df['career_diff'] = df['career_level_user'] -df['career_level_item']
    inter_df = df[['jobrole_list','title_list','tag_list']]
    inter_df['jobrole_list'] =inter_df['jobrole_list'].apply(set)
    inter_df['title_list'] =inter_df['title_list'].apply(set)
    inter_df['tag_list'] =inter_df['tag_list'].apply(set)
    df['job_title'] = inter_df.apply( lambda x: len(x['jobrole_list'].intersection(x['title_list'])),axis=1)
    df['job_tag'] = inter_df.apply( lambda x: len(x['jobrole_list'].intersection(x['tag_list'])),axis=1)
    return df

def intersection_similarity(matA,matB):
    intersection = matA.dot(matB.T)
    return intersection

def title_tag_sparse(df_items,item_to_col,tags_elements_dict,row_shape=None,col_shape=None):

    df_item_title = df_items[['item_id','title_list']].explode('title_list')
    df_item_title['value'] =1
    #user_to_rowtitle,item_to_coltitle =id_to_index(item_title,'id','title_list')
    df_item_title_sparse = sparcify(df_item_title,'item_id','title_list','value',item_to_col,tags_elements_dict,row_shape,col_shape)
    df_item_tag = df_items[['item_id','tag_list']].explode('tag_list')
    df_item_tag['value'] =1
    #user_to_rowtag,item_to_coltag =id_to_index(item_title,'id','tag_list')
    df_item_tag_sparse = sparcify(df_item_tag,'item_id','tag_list','value',item_to_col,tags_elements_dict,row_shape,col_shape)
    return df_item_title_sparse,df_item_tag_sparse

def percentage_contained(df,columnA,columnB):

  df[columnA] = df[columnA].fillna("").apply(list)
  # we can compute these in-place, no need to create new DataFrame columns
  intersect = [elem.count(val) for elem, val in zip(df[columnA], df[columnB])]
  a_length = [len(elem) for elem in df[columnA]]

  return  [round(i/max(j,1),3) for i, j in zip(intersect, a_length)]

def intu_item_similarity(test_user_ids,test_items,Intu,user_items):
    for row_num,ele in enumerate(test_user_ids):
        intu_mode_career_level = Intu.loc[ele]['intu_mode_career_level']
        intu_mode_industry_id =  Intu.loc[ele]['intu_mode_industry_id']
        intu_mode_discipline_id =  Intu.loc[ele]['intu_mode_discipline_id']
        item_id_list = test_items[(test_items['career_level'].isin(intu_mode_career_level)) 
                                    & (test_items['industry_id'].isin(intu_mode_industry_id)) & 
                                    (test_items['discipline_id'].isin(intu_mode_discipline_id))]['item_id'].tolist()

        user_items['item_id'].loc[ele]=user_items['item_id'].loc[ele] + item_id_list   
    return user_items