import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
from collections import OrderedDict as od
import argparse
import itertools
from collections import Counter
import math
import yaml

parser = argparse.ArgumentParser(description='Pass Files')
parser.add_argument('--interactions_file')
parser.add_argument('--users_file')
parser.add_argument('--items_file')
parser.add_argument('--files_save_path')
parser.add_argument('--data_config_file',default='data.yml')
args = parser.parse_args()


folder = args.files_save_path
test_interaction_file = folder + '/test_interaction.csv'
test_users_file = folder +'/test_users.csv'
train_users_file = folder + '/train_users.csv'
#assigned gender file
assigned_users_file = folder + '/assigned_users.csv'
unbiased_test_interactions_file = folder + '/unbiased_test_interactions.csv'
biased_test_interactions_file = folder + '/biased_test_interactions.csv'
train_interactions_file = folder + '/train_interactions.csv'
full_interactions =  args.interactions_file
# minimum interactions per user
lower_num_interaction = 20
upper_num_interaction =40
unbiased_count = 1000
#number of users for unbiased dataset
biased_count = 1000
rows_per_category = 20
user_key = 'user_id'
item_key = 'item_id'
timestamp_key = 'created_at'
trimmed_data = True

def load_data():
    items = pd.read_csv(args.items_file,sep='\t')
    users = pd.read_csv(args.users_file,sep='\t',dtype={'career_level':pd.Int64Dtype(), 'discipline_id':pd.Int64Dtype(), 'industry_id':pd.Int64Dtype(), 'experience_n_entries_class': pd.Int64Dtype(), 'experience_years_experience':pd.Int64Dtype(), 'experience_years_in_current':pd.Int64Dtype(),'edu_degree':pd.Int64Dtype()})
    interactions = pd.read_csv(args.interactions_file,sep = '\t')
    users = users.rename(columns={"id": "user_id"})
    items = items.rename(columns={"id": "item_id"}, errors="raise")
    users=users.drop_duplicates(subset=['user_id'])
    items=items.drop_duplicates(subset=['item_id'])
    users.fillna(100,inplace=True)
    items.fillna(-100,inplace=True)
    return items,users,interactions


def attribute_assignment(data,field_dict,set_attr, proportion, choices):
    """
    data: pandas dataframe which is used to assigned the attribute
    field_dict : fields based on which the attribute has to be assigned. fields are loaded from the config file 
    """

    df = data.copy()
    field_names = field_dict.keys()
    x=[{field : df[field].unique()} for field in field_names]
    combination = list(itertools.product(*list(field_dict.values()))) 
    for c in combination:
      new_ind = None
      for id,field in enumerate(field_names):
            ind = df[field] == c[id]
            if type(new_ind) != type(None):
               new_ind = ind & new_ind
            else:
                  new_ind =ind 
      if sum(new_ind)>0:
        total = sum(new_ind)
        choice_prop= list(zip(choices,proportion))
        final_array=[]
        for item in choice_prop:
          element = item[0]
          prop=item[1]
          arr = [element] * math.ceil(total*prop)
          final_array = final_array+arr
        final_array =np.array(final_array[:total])
        np.random.shuffle(final_array)
        df.loc[new_ind, set_attr] = final_array
    
   
    return df

def filter_users(interactions,users,items,lower_num_interaction,upper_num_interaction,field_filter =[('industry_id',[1,8,0,21])]):
    #users with atleast 3 interactions
    interactions = interactions[interactions[item_key].isin(items[item_key].unique())]
    interactions = interactions[interactions[user_key].isin(users[user_key].unique())]
    interactions_positive = interactions[interactions['interaction_type'] != 4]
    users_pop = interactions_positive.user_id.value_counts()
    good_users = users_pop[(users_pop >=lower_num_interaction) & (users_pop <=upper_num_interaction)].index

    if trimmed_data:
        #users only from 1,8,0,21 industry
        for filter_name,values in field_filter:
            limited_users = users[users[filter_name].isin(values)]
            limited_users = limited_users[limited_users[user_key].isin(good_users)]
    else:
        limited_users = users[users[user_key].isin(good_users)]
    limited_users = limited_users.groupby('industry_id',as_index=False).apply(lambda x: 
    x.sample(2450,replace=False,random_state =1))

    limited_users.reset_index(inplace=True,drop=True)
    
    interactions = interactions[interactions.user_id.isin(limited_users[user_key].values)]
    return interactions,limited_users


def seperate_users(limited_users,users,rows_per_category,full_interactions,interactions,fields ):
    high_career_level = limited_users[limited_users['career_level'].isin([5,6,7])]

   #limited_users = pd.concat([high_career_level])
    unbiased_test_users = limited_users.groupby(fields,as_index=False).apply(lambda x: 
    x.sample(rows_per_category,replace=True,random_state =1))
    unbiased_test_users.drop_duplicates(subset=[user_key],inplace=True)
    unbiased_test_users =unbiased_test_users.sample(unbiased_count)
    train_users = limited_users[~limited_users[user_key].isin(unbiased_test_users[user_key].unique())]
    unbiased_test_users.reset_index(inplace=True,drop=True)
    
    biasedtest_users = train_users.groupby('industry_id',as_index=False).apply(lambda x: 
    x.sample(250,replace=False,random_state =1))#.sample(biased_count,random_state=1)
    biasedtest_users.drop_duplicates(subset=[user_key],inplace=True)
    biasedtest_users.reset_index(inplace=True,drop=True)
    train_users = train_users[~train_users[user_key].isin(biasedtest_users[user_key])]
    interactions_positive = interactions[interactions['interaction_type'] != 4]
    unbiased_test_interaction = interactions_positive[interactions_positive[user_key]
    .isin(unbiased_test_users[user_key])].sort_values([timestamp_key],ascending=False).groupby([user_key]
    ,as_index=False).apply(variable_head)
    # split data by taking latest interaction as test data for each user id
    biased_test_interaction = interactions_positive[interactions_positive[user_key]
    .isin(biasedtest_users[user_key])].sort_values([timestamp_key],
    ascending=False).groupby([user_key],as_index=False).apply(variable_head)
    # removing biased and unbiased interaction
    train_interaction = interactions[~interactions.index.isin(unbiased_test_interaction.droplevel(0).index)]
    train_interaction = train_interaction[~train_interaction.index.isin(biased_test_interaction.droplevel(0).index)]
    negative_interaction_ids = train_interaction[train_interaction['interaction_type'] == 4][item_key].to_list()  
    #cold start items are those items which 
    cold_start_item_interactions = interactions_positive[interactions_positive[item_key]
    .isin(biased_test_interaction.item_id.tolist()+unbiased_test_interaction.item_id.to_list()+ negative_interaction_ids)]
    positive_train_interaction = train_interaction[train_interaction['interaction_type'] !=4]
    full_interactions_positive = full_interactions[full_interactions['interaction_type'] != 4]
    cold_start_item_interactions = cold_start_item_interactions[~cold_start_item_interactions[item_key]
    .isin(positive_train_interaction.item_id.tolist())]
    full_interactions_positive = full_interactions_positive[full_interactions_positive['item_id'].isin(cold_start_item_interactions['item_id'].values)]
    #full_interactions_positive =full_interactions_positive[~full_interactions_positive[user_key]
    #.isin(biasedtest_users.user_id.to_list()+unbiased_test_users.user_id.to_list())]
    cold_start_item_interactions = full_interactions_positive.merge(users,how='left',left_on=user_key,right_on=user_key)
    cold_start_item_interactions = cold_start_item_interactions[cold_start_item_interactions['industry_id'].isin([0,21,1,8])]
    cold_start_item_interactions = cold_start_item_interactions[~cold_start_item_interactions['user_id'].isin(biasedtest_users['user_id'].to_list()+unbiased_test_users['user_id'].to_list())]
    cold_start_item_interactions = cold_start_item_interactions.sort_values(by=timestamp_key
    ,ascending=False).groupby(item_key).head(1)
    cold_start_item_interactions = cold_start_item_interactions[['user_id','item_id','created_at','interaction_type']]
    cold_start_users = users[users['user_id'].isin(cold_start_item_interactions['user_id'].unique())]

    train_interaction = pd.concat([train_interaction,cold_start_item_interactions])
    train_users = pd.concat([train_users,cold_start_users])
    train_users = train_users.drop_duplicates(user_key)

    return train_interaction,unbiased_test_interaction,biased_test_interaction,train_users,biasedtest_users,unbiased_test_users

    

def variable_head(group,head_size=10):
    return group.head(head_size)


def attrib_assignment_util(test_users,train_users,unbiased_test_users):
    with open(args.data_config_file, 'r') as file:
       config = yaml.safe_load(file)
    for conf in config['config_biased']:
         subconfig =config['config_biased']
         train_users = attribute_assignment(train_users,field_dict = od(subconfig[conf]['fields']), set_attr = subconfig[conf]['set_attr'], 
         proportion =subconfig[conf]['proportion'], choices = subconfig[conf]['choices'])
         test_users = attribute_assignment(test_users,field_dict = od(subconfig[conf]['fields']), set_attr = subconfig[conf]['set_attr'], 
         proportion = subconfig[conf]['proportion'], choices = subconfig[conf]['choices'])
    for conf in config['config_unbiased']:   
        subconfig =config['config_unbiased']  
        unbiased_users_test = attribute_assignment(unbiased_test_users,field_dict = od(subconfig[conf]['fields']),
        set_attr = subconfig[conf]['set_attr'], proportion = subconfig[conf]['proportion'], choices = subconfig[conf]['choices'])
    all_users = pd.concat([test_users,train_users])
    return all_users,unbiased_users_test

def save(all_interactions, all_users,unbiased_users_test,unbiased_test_interaction,biased_test_interaction,train_interaction):
    all_interactions.to_csv(test_interaction_file,header=True,index=False)
    pd.concat([all_users,unbiased_users_test]).to_csv(assigned_users_file,header=True,index=False)
    unbiased_test_interaction.to_csv(unbiased_test_interactions_file,header=True,index=False)
    biased_test_interaction.to_csv(biased_test_interactions_file,header=True,index=False)
    train_interaction.to_csv(train_interactions_file,header=True)

def main():
    items,users,interactions = load_data()
    filtered_interactions,limited_users = filter_users(interactions,users,items,lower_num_interaction,upper_num_interaction)
    print('read..files')
    train_interaction,unbiased_test_interaction,biased_test_interaction,train_users,biasedtest_users,unbiased_test_users = seperate_users(limited_users,users,rows_per_category,interactions,filtered_interactions,fields =['career_level','experience_years_experience','industry_id'])
    print('iteractions..files')     
    all_users,unbiased_users_test = attrib_assignment_util(biasedtest_users,train_users,unbiased_test_users)
    print('attribute')
    all_interactions = pd.concat([biased_test_interaction,unbiased_test_interaction])
    save(all_interactions, all_users,unbiased_users_test,unbiased_test_interaction,biased_test_interaction,train_interaction)

if __name__ == "__main__":
    main() 