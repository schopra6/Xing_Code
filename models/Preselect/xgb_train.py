from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
import pandas as pd
from  metrics.metrics import challenge_score,compute_metrics,totalRelevant,GCE
from statistics import median,mean,mode
import argparse
features = [  'interaction_type', 'experience_n_entries_class',
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
       'intuser_experience_years_in_current_contained','intuser_gender_contained']

def left_out(Y,X):
       diff=[]
       selected=[]
       for uid in X.index:
              diff.append(len(set(Y.loc[uid].values[0]) - set(X[uid])))
              selected.append(len(set(X[uid])))
       print(f"mean {mean(diff)} median {median(diff)} and mode {mode(diff)}")
       print(f"mean{mean(selected)} median {median(selected)}")
def run(folder,train_data,test_data,test_interactions):       
       data =train_data[features]

       data['interaction_type'] = ((data['interaction_type']==1) | (data['interaction_type']==2) | (data['interaction_type']==3)).astype(int)
       X_train, X_test, y_train, y_test = train_test_split(data.drop('interaction_type',axis=1), data['interaction_type'], test_size=.01)
       bst = XGBClassifier(n_estimators=15,n_jobs=-1, max_depth=6, learning_rate=0.1,gamma=1,min_child_weight=4,objective='binary:logistic')
       #bst2 = XGBClassifier(n_estimators=100,n_jobs=100, max_depth=6, learning_rate=0.1,gamma=1,min_child_weight=4,gpu_id=6,tree_method='gpu_hist',objective='binary:logistic')
       bst.fit(X_train, y_train)
       from sklearn.metrics import accuracy_score
       # make predictions for test data
       y_prediction = bst.predict(X_test)
       print(len(y_prediction[y_prediction==0]))
       predictions = [round(value) for value in y_prediction]
       # evaluate predictions
       accuracy = accuracy_score(y_test, predictions)
       common_df = pd.merge(test_data,test_interactions[['user_id','item_id','interaction_type']],how = 'left',on=['user_id', 'item_id'])
       common_df['interaction_type'] = common_df['interaction_type'].fillna(0)
       common_df['interaction_type'] = ((common_df['interaction_type']==1) | (common_df['interaction_type']==2) | (common_df['interaction_type']==3)).astype(int)
       y_prediction_test = bst.predict(common_df[features].drop('interaction_type',axis=1))
       print(precision_score(common_df['interaction_type'],y_prediction_test))
       features.remove('interaction_type')
       data_test=test_data[features]

      
       test_data['y_pred'] = bst.predict_proba(data_test)[:,1]
       sorted_items = test_data[['user_id','item_id','y_pred']].sort_values(by=['y_pred'], ascending=[False]).groupby('user_id').head(30)
       recommended_users = sorted_items.groupby('user_id')['item_id'].agg(list).reset_index()
       Y_labels =test_interactions[['user_id','item_id']].groupby('user_id')['item_id'].agg(list).reset_index().set_index('user_id')
       print(left_out(Y_labels,test_data[['user_id','item_id']].groupby('user_id')['item_id'].agg(list)))
       recommended_users.to_csv(folder + '/results.csv',index=False)
       

       print(compute_metrics(recommended_users.set_index('user_id'),Y_labels))

def main():   

       parser = argparse.ArgumentParser(description='Pass Files')
       parser.add_argument('--files_read_path')
       args = parser.parse_args()
       folder = args.files_read_path
       train_data = pd.read_csv(folder+'/train_data.csv')
       test_data = pd.read_csv(folder+'/test_data.csv')
       test_interactions = pd.read_csv(folder+'/test_interaction.csv')
       run(folder,train_data,test_data,test_interactions)

if __name__ == "__main__":

    main()     