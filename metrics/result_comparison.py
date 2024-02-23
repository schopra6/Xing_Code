import pandas as pd
import glob
import os
import numpy as np
from  metrics.metrics import challenge_score,compute_metrics,totalRelevant,get_p,totalRelevantcount,GCE,return_metrics
import argparse
from collections import Counter
from scipy import stats

def attrib_df(df,attrib,attrib_value):
  return df[df[attrib].isin(attrib_value)]

def get_dict(df,attrib_set):
    attrib_set_keys = attrib_set.keys()
    df_dict={}
    for key in attrib_set_keys:
        for sub_key,value in attrib_set[key].items():
            df_dict[sub_key] = attrib_df(df,key,value)
    return df_dict

def print_attrib_results(typeof,df_dict,attrib_set,relavent):
    attrib_set_keys = attrib_set.keys()
    count = {}
    for key in attrib_set_keys:
        results={}
        mean_results={}
        mincount = 0
        for sub_key,value in attrib_set[key].items():
         if mincount == 0:
             mincount= len(df_dict[sub_key])
         else  :  
           mincount= min(mincount,len(df_dict[sub_key]))
        for sub_key,value in attrib_set[key].items():
            print(f"{typeof},{sub_key}, {compute_metrics(df_dict[sub_key].sample(mincount,random_state=30)[['user_id','item_id']].set_index('user_id'),relavent)}")
            count[sub_key] =totalRelevantcount(df_dict[sub_key].sample(mincount,random_state=30)[['user_id','item_id']].set_index('user_id'),relavent) 
            results[sub_key] = return_metrics(df_dict[sub_key].sample(mincount,random_state=30)[['user_id','item_id']].set_index('user_id'),relavent) 
            mean_results[sub_key] = compute_metrics(df_dict[sub_key].sample(mincount,random_state=30)[['user_id','item_id']].set_index('user_id'),relavent)
        subkeys = sorted(attrib_set[key].keys())
        #print(mean_results[subkeys[0]])
        print(subkeys)
        res = tuple(round((ele1 -ele2)*100/ ele2,3) for ele1, ele2 in zip(mean_results[subkeys[0]], mean_results[subkeys[1]]))
        print(f",,{res}")
        #print(f"{key},{subkeys[0]},{subkeys[1]},",end=',')
        #for i in range(6):
        #    print(f"{stats.ttest_ind(results[subkeys[0]][i],results[subkeys[1]][i],alternative='less').pvalue}",end=',',sep = ' ')
        #print()
        #gce = GCE(attribute=attrib_set[key].keys(),p = get_p(attrib_set[key].keys(),count),pf ={'male':1/2,'female':1/2},alpha=-1) 
        #print(f"{typeof},{gce}")      

def print_attrib_proportion(typeof,df_dict,attrib_set,features):
    attrib_set_keys = attrib_set.keys()
    count = {}
    for key in attrib_set_keys:
        for sub_key,value in attrib_set[key].items():
            for feature in features:
             print(f"{typeof},{sub_key},{feature},{len(df_dict[sub_key])},{compute_feature_prop(df_dict[sub_key],feature)}")

        #gce = GCE(attribute=attrib_set[key].keys(),p = get_p(attrib_set[key].keys(),count),pf ={'male':1/2,'female':1/2},alpha=-1) 
        #print(f"{typeof},{gce}")      

def compute_feature_prop(df,feature):
   cnt = Counter(df[feature].values)
   total = sum(cnt.values())
   cnt = { key:round(value/total,5) for key,value in cnt.items()}
   return cnt

def run_result_comparison(assigned_users,items,unbiased_test_interactions,biased_test_interactions,recommended,test_interactions):
    unbiased_users = unbiased_test_interactions['user_id'].unique()
    biased_users = biased_test_interactions['user_id'].unique()
    df_unbiased = assigned_users[assigned_users['user_id'].isin(unbiased_users)]
    df_biased = assigned_users[assigned_users['user_id'].isin(biased_users)]
    df_unbiased = df_unbiased.merge(recommended, on='user_id',how='inner')
    df_biased = df_biased.merge(recommended, on='user_id',how='inner')
    df_unbiased['item_id'] = df_unbiased['item_id'].apply(eval)
    df_biased['item_id'] = df_biased['item_id'].apply(eval)
    item_bias = df_biased.explode('item_id')
    item_unbias  = df_unbiased.explode('item_id')
    item_bias  = item_bias[['user_id','item_id','gender','industry_id','career_level','experience_years_experience']].merge(items, on='item_id',how='inner', suffixes=('_user', '_item'))
    item_unbias  = item_unbias[['user_id','item_id','gender','industry_id','career_level','experience_years_experience']].merge(items, on='item_id',how='inner', suffixes=('_user', '_item'))
    attrib_set ={'gender':{'male':[1],'female':[0]},'industry_id':{'female_dominated':[1,8],'male_dominated':[0,21]}}
    attrib_set_merged ={'gender':{'male':[1],'female':[0]},'industry_id_user':{'female_dominated':[1,8],'male_dominated':[0,21]},'career_level_user':{'high':[4,5,6,7,8],'low':[0,1,2,3]},'experience_years_experience':{'high':[4,5,6,7,8],'low':[0,1,2,3]}}
    attrib_set_gender ={'gender':{'male':[1],'female':[0]}}
    unbiased_dict = get_dict(df_unbiased,attrib_set)
    biased_dict = get_dict(df_biased,attrib_set)
    male_biased_dict = get_dict(biased_dict['male_dominated'],attrib_set_gender)
    male_unbiased_dict = get_dict(unbiased_dict['male_dominated'],attrib_set_gender)
    female_biased_dict = get_dict(biased_dict['female_dominated'],attrib_set_gender)
    female_unbiased_dict = get_dict(unbiased_dict['female_dominated'],attrib_set_gender)
    #################################################
    unbiased_item_dict = get_dict(item_unbias,attrib_set_merged)
    biased_item_dict = get_dict(item_bias,attrib_set_merged)
    high_biased_item_dict = get_dict(biased_item_dict['high'],attrib_set_merged)
    high_unbiased_item_dict = get_dict(unbiased_item_dict['high'],attrib_set_merged)
    high_biased_item_dict_male = get_dict(high_biased_item_dict['male_dominated'],attrib_set_merged)
    high_unbiased_item_dict_male = get_dict(high_unbiased_item_dict['male_dominated'],attrib_set_merged)
    #print_attrib_proportion('unbiased',high_unbiased_item_dict_male,attrib_set_gender,['career_level_item'])
    #print_attrib_proportion('biased',high_biased_item_dict_male,attrib_set_gender,['career_level_item'])
    relavent = test_interactions.groupby('user_id')['item_id'].agg(list)
    relavent = pd.DataFrame(relavent)
    print_attrib_results('unbiased_male_0_21',male_unbiased_dict,attrib_set_gender,relavent)
    print_attrib_results('biased_male_0_21',male_biased_dict,attrib_set_gender,relavent)
    print_attrib_results('unbiased_female_1_8',female_unbiased_dict,attrib_set_gender,relavent)
    print_attrib_results('biased_female_1_8',female_biased_dict,attrib_set_gender,relavent)


def main():   

       parser = argparse.ArgumentParser(description='Pass Files')
       parser.add_argument('--files_read_path')
       args = parser.parse_args()
       folder = args.files_read_path
       items_file =  '../items.csv'
       assigned_users_file = folder + '/assigned_users.csv'
       unbiased_test_interactions_file = folder + '/unbiased_test_interactions.csv'
       biased_test_interactions_file = folder + '/biased_test_interactions.csv'
       test_interactions_file = folder + '/test_interaction.csv'
       recommended_interaction_file =folder + '/results.csv'
       assigned_users = pd.read_csv(assigned_users_file)
       items = pd.read_csv(items_file,sep='\t',dtype={'id':pd.Int64Dtype(),'career_level':pd.Int64Dtype(), 'discipline_id':pd.Int64Dtype(), 'industry_id':pd.Int64Dtype()})
       assigned_users.drop_duplicates(subset=['user_id'],inplace=True)
       items = items.rename(columns={"id": "item_id"}, errors="raise")
       items.drop_duplicates(subset=['item_id'],inplace=True)
       test_interactions = pd.read_csv(test_interactions_file)
       unbiased_test_interactions = pd.read_csv(unbiased_test_interactions_file)
       biased_test_interactions = pd.read_csv(biased_test_interactions_file)
       recommended = pd.read_csv(recommended_interaction_file)
       run_result_comparison(assigned_users,items,unbiased_test_interactions,biased_test_interactions,recommended,test_interactions)

if __name__ == "__main__":

    main() 