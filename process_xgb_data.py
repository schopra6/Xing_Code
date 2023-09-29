from statistics import median,mean,mode
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
import argparse
import scipy
from scipy.sparse import csr_matrix
user_key = 'user_id'
item_key = 'item_id'
from create_features import *
from xgb_train import *

timestamp_key = 'created_at'
trimmed_data = False
jaccard_threshold = 0.5
sample_size=10
#'intu_tag_list_item_contained','intu_title_list_item_contained','intuser_jobrole_list_contained', 
features = ['user_id','item_id', 'interaction_type', 'experience_n_entries_class',
       'experience_years_experience', 'experience_years_in_current',
       'edu_degree', 'latitude',
       'longitude','gender',
       'clicks','prev_count',
       'item_id_click_ratio', 'item_id_max_time', 'career_diff', 'item_user_max_time',
       'intu_career_level_item_contained', 'intu_region_item_contained',
       'intu_industry_id_item_contained', 'intu_discipline_id_item_contained',
       'intu_country_item_contained','intuser_career_level_user_contained',
       'intuser_region_user_contained', 'intuser_industry_id_user_contained',
       'intuser_discipline_id_user_contained',
       'intuser_country_user_contained',
       'intuser_experience_years_experience_contained',
       'intuser_edu_degree_contained',
       'intuser_experience_years_in_current_contained']

def load_file(args,assigned_users_file,train_interactions_file,test_interaction_file):
    items = pd.read_csv(args.items_file,sep='\t')
    assigned_users = pd.read_csv(assigned_users_file)
    items.fillna(-100,inplace=True)
    items = items.rename(columns={"id": "item_id"}, errors="raise")
    assigned_users = assigned_users.rename(columns={"id": "user_id"})
    test_interaction = pd.read_csv(test_interaction_file)
    train_interactions = pd.read_csv(train_interactions_file)
    assigned_users.fillna(100,inplace=True)
    assigned_users=assigned_users.drop_duplicates(subset = [user_key])
    items=items.drop_duplicates(subset = [item_key])
    return items,test_interaction,train_interactions,assigned_users

def missed_item_stats(recommended_list, groundtruth_list,test_user_ids):
  left_out_ids=[]  
  for user in test_user_ids:
    left_out_ids.append(len(set(groundtruth_list.loc[user][item_key]) - set(groundtruth_list.loc[user][item_key])))
  print(f"mean {mean(left_out_ids)} median {median(left_out_ids)} and mode {mode(left_out_ids)}")

def item_process(items):
    items['title_list'] = items['title'].apply(lambda x: [int(ele ) for ele in str(x).split(",")])
    items['tag_list'] = items['tags'].apply(lambda x: [int(ele ) for ele in str(x).split(",")])
    #train_items =positive_items[positive_items['id'].isin(dropped_interactions_train[item_key].unique())]
    #test_items =positive_items[positive_items['id'].isin(dropped_interactions_test[item_key].unique())]
    #items =items[items[user_key].isin(train_interactions[item_key].unique())] 
    return items   

def user_process(assigned_users):
     assigned_users['jobrole_list'] = assigned_users['jobroles'].apply(lambda x: [int(ele ) if len(str(x)) > 0 and str(x) != 'nan' else -100 for ele in str(x).split(",")])
     return assigned_users


def negative_sampling(train_interactions,user_key,item_key,sample_size=10,feedback_key='interaction_type',timestamp_key='created_at',timestamp=None):
    unique_users = train_interactions[user_key].drop_duplicates()
    unique_items = train_interactions[item_key].unique()
    non_interacted_pairs=unique_users.map(lambda user:  (user,set(random.sample(unique_items.tolist(),sample_size)) 
                                                         - set(train_interactions[train_interactions[user_key] == user][item_key])))
    non_interacted_df = pd.DataFrame(non_interacted_pairs.to_list(), columns=[user_key,item_key])
    non_interacted_df[item_key]= non_interacted_df[item_key].apply(list)
    non_interacted_df =non_interacted_df.explode(item_key).reset_index()
    non_interacted_df[feedback_key] =4
    non_interacted_df[timestamp_key] =timestamp
    return non_interacted_df

def train_test_data(train_data,test_data,users,items,test_user_ids,test_item_ids,positive_interactions,user_key='user_id',item_key=item_key):
    test_data = test_data.drop_duplicates(subset =[user_key,item_key])
    train_data=train_data.merge(users,left_on=user_key,right_on=user_key)
    train_data =train_data.merge(items,left_on=item_key,right_on=item_key,suffixes=('_user', '_item'))
    test_data=test_data.merge(users,left_on=user_key,right_on=user_key)
    test_data=test_data.merge(items,left_on=item_key,right_on=item_key,suffixes=('_user', '_item'))
    train_data = event_feature(train_data)
    test_data = event_feature(test_data)
    train_data = content_similarity(train_data)
    test_data = content_similarity(test_data)
    train_data = last_click_activity(train_data)
    test_data = last_click_activity(test_data)
    train_data = previous_click(positive_interactions,train_data)
    test_data = previous_click(positive_interactions,test_data)



    return train_data,test_data

def sparcify(df,row_name,col_name,value_name,user_to_row,item_to_col,row_shape=None,col_shape=None,):

    # Create a sparse matrix directly from data
    if not row_shape:
        row_shape = len(user_to_row)
    if not col_shape:    
        col_shape = len(item_to_col)

    # Initialize arrays to store rows, columns, and values for the sparse matrix
    rows = np.array([user_to_row[user_id] for user_id in df[row_name]])
    cols = np.array([item_to_col[item_id] for item_id in df[col_name]])
    values = np.repeat(1,len(rows))
    # Create the sparse matrix
    sparse_collaborative_matrix = csr_matrix((values, (rows, cols)), shape=(row_shape, col_shape))
    return sparse_collaborative_matrix

    
def id_to_index(df):
    item_to_col = {item_id: i for i, item_id in enumerate(df)}
    return item_to_col



def item_postprocess(items,merge_dfs,item_key=item_key):
    for merge_df in merge_dfs:
       items=items.merge(merge_df,left_on=item_key,right_on=item_key,how='left')
    return(items)   

def user_postprocess(users,merge_dfs,user_key='user_id'):
    for merge_df in merge_dfs:
        users=users.merge(merge_df,left_on=user_key,right_on=user_key,how='left')
    return(users)

def id_utils(train_interactions,test_interactions):
    train_item_ids = train_interactions[item_key].unique()
    test_item_ids = test_interactions[item_key].unique()
    train_user_ids = train_interactions[user_key].unique()
    test_user_ids = test_interactions[user_key].unique()
    return train_item_ids,test_item_ids,train_user_ids,test_user_ids
   

def main():
    parser = argparse.ArgumentParser(description='Pass Files')
    parser.add_argument('--items_file', default='item')
    parser.add_argument('--files_read_path', default='folder')
    args = parser.parse_args()



    folder = args.files_read_path
    test_interaction_file = folder + '/test_interaction.csv'
    #assigned gender file
    assigned_users_file = folder + '/assigned_users.csv'
    train_interactions_file = folder + '/train_interactions.csv'

    items,test_interactions,train_interactions,assigned_users = load_file(args,assigned_users_file,train_interactions_file,test_interaction_file)
    items = item_process(items)
    assigned_users = user_process(assigned_users)
    negative_interactions = negative_sampling(train_interactions,user_key='user_id',item_key=item_key,sample_size=10,timestamp=train_interactions.created_at.iloc[0])
    negative_interactions = negative_interactions.drop('index',axis=1)
    train_interactions = pd.concat([train_interactions,negative_interactions])
    train_interactions['created_at'] = pd.to_datetime(train_interactions['created_at'], unit='s')
    train_interactions['week'] = train_interactions.created_at.dt.strftime('%W')
    positive_train_interactions = train_interactions[train_interactions['interaction_type']!=4]
    item_clicks = clicks(positive_train_interactions)
    click_ratio_df = click_ratio(positive_train_interactions)
    user_events = user_activity(train_interactions)
    item_id_max,user_id_max = click_date_max(train_interactions)
    train_item_ids, test_item_ids, train_user_ids,  test_user_ids =id_utils(train_interactions,test_interactions)
    item_ids = np.unique(np.append(test_item_ids,train_item_ids))
    items = items[items[item_key].isin(item_ids)]
    intu,intuser = Intu_Intusers(positive_train_interactions,assigned_users.set_index(user_key),items.set_index(item_key))
    user_item = intu[[user_key,item_key]].set_index(user_key)
    intu = intu.drop(item_key,axis=1)

    items = item_postprocess(items,[intuser,item_clicks,click_ratio_df,item_id_max])
    assigned_users = user_postprocess(assigned_users,[intu,user_events,user_id_max])



    train_items = items[items[item_key].isin(train_item_ids)]
    test_items = items[items[item_key].isin(test_item_ids)]
    user_item = intu_item_similarity(test_user_ids,test_items,intu.set_index(user_key),user_item)
    train_users = assigned_users[assigned_users[user_key].isin(train_user_ids)]
    test_users = assigned_users[assigned_users[user_key].isin(test_user_ids)]
    user_to_row = {user_id: i for i, user_id in enumerate(assigned_users[user_key].unique())}
    item_to_col = {item_id: i for i, item_id in enumerate(item_ids)}
    jobrole_list = sum(assigned_users.jobrole_list, [])
    tag_list = sum(items.tag_list, [])
    title_list = sum(items.title_list, [])
    item_tags = set(jobrole_list + tag_list +title_list)
    tags_elements_dict =id_to_index(item_tags)
    
    user_jobrole = test_users[[user_key,'jobrole_list']].explode('jobrole_list')
    user_jobrole['value'] =1
    user_jobrole_sparse = sparcify(user_jobrole,user_key,'jobrole_list','value',user_to_row,tags_elements_dict)

    train_item_title_sparse,train_item_tag_sparse = title_tag_sparse(train_items,item_to_col,tags_elements_dict)
    test_item_title_sparse,test_item_tag_sparse = title_tag_sparse(test_items,item_to_col,tags_elements_dict)
    testuser_i =[ user_to_row[id] for id in test_user_ids]
    testuser_i = np.array(testuser_i)
    sparse_collaborative_matrix = sparcify(positive_train_interactions.drop_duplicates(subset=[user_key,item_key]),user_key,item_key,'interaction_type',user_to_row,item_to_col)
    testuser_mat= sparse_collaborative_matrix[testuser_i,:]
    jaccard = jaccard_similarity(testuser_mat,sparse_collaborative_matrix)
    user_item =jaccard_util(jaccard,test_user_ids,user_item,jaccard_threshold,item_ids,sparse_collaborative_matrix)
    user_item = similarity_util(sparse_collaborative_matrix,test_user_ids,train_item_title_sparse,test_item_title_sparse,train_item_tag_sparse,test_item_tag_sparse,user_jobrole_sparse,user_item,user_to_row,item_ids,test_item_ids)
    user_item = user_item[user_item.index.isin(test_user_ids)]
    popular_items = list(item_clicks.sort_values(by='clicks',ascending=False)['item_id'].values[:30])
    user_item['item_id'] = user_item['item_id'].map(lambda x : x +popular_items)
    test_data = user_item.reset_index().explode([item_key])
    test_data = test_data.drop_duplicates(subset =[user_key,item_key])
    
    train_data,test_data = train_test_data(train_interactions,test_data,assigned_users,items,test_user_ids,test_item_ids,positive_train_interactions,user_key=user_key,item_key=item_key)
    train_data=train_data[features]
    run(folder,train_data,test_data,test_interactions)

    train_data.to_csv(folder+'/train_data.csv',index=False)
    features.remove('interaction_type')
    test_data =test_data[features]
    test_data.to_csv(folder+'/test_data.csv',index=False)

if __name__ == "__main__":

    main()     