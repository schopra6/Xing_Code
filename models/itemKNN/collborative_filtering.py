from scipy.cluster.vq import vq, kmeans2
from scipy import *
import datetime
import csv
import numpy as np
import scipy.sparse.linalg
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import multiprocessing
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
num_users = 1500001
num_items = 1358098
num_tests = 150000
num_clusters = 380

userid_to_index = {}
users_matrix = []
itemid_to_index = {}
unbiased_itemid_to_index = {}
biased_itemid_to_index = {}
items_matrix = []
clust_to_index = {}
items_delete_columns = [0, 1, 2,76,77]
users_delete_columns = [0, 1, 2]
interaction_matrix = scipy.sparse.lil_matrix((num_users, num_items))
users_with_clusters = []

def parse_args():
    parser = argparse.ArgumentParser(description='filter the dataset.')
    parser.add_argument('--interaction_path', type=str,
                        help='path to the dataset')
    parser.add_argument('--n_workers', type=int,default = 20) 
    parser.add_argument('--train', type=bool,default = False)                       
    parser.add_argument('--save_filename', type=str, default = 'data/CF_ItemItemSimilarity_biased_v3',
                        help='filename to save the  dataset')
    args = parser.parse_args()
    return args

def read_files(users_indicator_file = 'data/users_indicators.csv',items_indicator_file = 'data/items_indicators.csv',train_interaction_file = '../prepare_data/reduced_data/train_interactions.csv',test_interaction_file='../prepare_data/reduced_data/test_interaction.csv'):
    print("Reading users_indicators.csv file ...")
    users_rows = csv.reader(open(users_indicator_file), delimiter='\t')
    next(users_rows)



    #index_to_users = {v: k for k, v in userid_to_index.items()}
    print("Reading items_indicators.csv file ...")
    items_rows = csv.reader(open(items_indicator_file), delimiter='\t')
    next(items_rows)
    print("Reading interactions.csv file ...")
    train_interactions_reader = pd.read_csv(train_interaction_file)
    if test_interaction_file:
     test_interaction_reader = pd.read_csv(test_interaction_file)
    else:
        test_interaction_reader=None
    return users_rows,items_rows,train_interactions_reader,test_interaction_reader


def cosine(X,Y):
   X= normalize(X)
   Y= normalize(Y)
   X =sparse.csr_matrix(X) 
   Y=Y.T
   print('ok1')
   return X.dot(Y)



def generate_data(users_rows,items_rows):
    global users_matrix,items_matrix,users_with_clusters
    print("Generating user_matrix ...")
    users_matrix_array = np.asarray(users_matrix)
    users_matrix_array[users_matrix_array == ''] = '0'
    users_matrix=users_matrix_array
    print("Clustering user data ...")
    kmeans_users =KMeans(num_clusters, random_state=0).fit(np.delete(users_matrix_array, users_delete_columns, 1).astype(float))
    print("Appending clusters to last column of items_matrix ...")
    users_with_clusters = np.concatenate((users_matrix_array, kmeans_users.labels_.reshape(kmeans_users.labels_.shape + (1,))), 1)

    clust_to_users = scipy.sparse.lil_matrix((num_clusters, num_users))
    for clust in range(0, num_clusters):
        users_in_clust = users_with_clusters[users_with_clusters[:, - 1] == str(clust)][:, 1]
        for userid in users_in_clust:
            clust_to_users[clust, userid_to_index[str(userid)]] = 1
    clust_to_users_csr = clust_to_users.tocsr()

    item_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(item_time)

    print("Generating items_matrix ...")
    items_matrix_array = np.array(items_matrix)
    items_matrix_array[items_matrix_array == ''] = '0'
    items_matrix=items_matrix_array
    print("Clustering items data ...")
    kmeans = KMeans(num_clusters, random_state=0).fit(np.delete(items_matrix_array,items_delete_columns , 1).astype(float)) 

    print("Appending clusters to last column of items_matrix ...")
    items_with_clusters = np.concatenate((items_matrix_array, kmeans.labels_.reshape(kmeans.labels_.shape + (1,))), 1)

    clust_to_items = scipy.sparse.lil_matrix((num_clusters, num_items))
    for clust in range(0, num_clusters):
        items_in_clust = items_with_clusters[items_with_clusters[:, items_with_clusters.shape[1] - 1] ==str(clust)][:, 1]
        for itemid in items_in_clust:
            clust_to_items[clust, itemid_to_index[str(itemid)]] = 1
    clust_to_items_csr = clust_to_items.tocsr()

    intr_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(intr_time)

    
    return  clust_to_items_csr,clust_to_users_csr



def worker(test_user_ids,index_to_selected_items,clust_to_users_csr,clust_to_items_csr,interaction_matrix_csr,filename):
    items_matrix_processed = np.delete(items_matrix, items_delete_columns, 1).astype(float)
    #users_matrix_processed = np.delete(users_matrix, users_delete_columns, 1).astype(float)
    print("Opening similarity output file ...")
    with open(filename, 'w') as fpResult:
        num_row = 0
        
        for test_userid in test_user_ids:
            num_items_written = 0
            alpha_z = scipy.sparse.lil_matrix((1, num_items))
            alpha_z = alpha_z.todok()
            test_index = userid_to_index[test_userid]
            beta = interaction_matrix_csr[test_index, :]
            num_row += 1
            if beta.getnnz() == 0:
                    label = users_with_clusters[test_index,-1].astype(int)
                    similar_users = clust_to_users_csr[label, :].nonzero()[1]
                    beta = interaction_matrix_csr[similar_users, :]
                    idx = beta.nonzero()[1] 
                    alpha_z[0,idx] = beta.data             
            else:
                for clust in range (0, num_clusters):
                    alpha = clust_to_items_csr[clust, :]
                    alpha_beta = beta.multiply(alpha)
                    ab_items =  items_matrix_processed[alpha_beta.indices]
                    alpha_items =  items_matrix_processed[alpha.indices]
                    if ab_items.shape[0] != 0:
                        similar_items = 1-cdist(ab_items ,alpha_items)
                        weighted_similar = np.mean(similar_items,axis=0)
                        alpha_z[alpha.nonzero()] =weighted_similar
                idx = beta.nonzero()
                alpha_z[idx] = beta.data
            
            jobs_score = alpha_z.toarray()
            final_vector = np.argsort(jobs_score)[:, ::-1][0]
            final_vector = final_vector[np.in1d(final_vector, np.array(list(index_to_selected_items.keys())), assume_unique=True)]
            tobeWritten = str(test_userid) + '\t'
            for item_index in final_vector:
                item_id = index_to_selected_items[int(item_index)]
                tobeWritten += str(item_id)
                num_items_written += 1
                if num_items_written > 30:
                        break
                tobeWritten += ','
            fpResult.write(tobeWritten + '\n')


        #end2_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        #print(end2_time)

def main():
    args = parse_args()
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time)
    users_rows,items_rows,train_interactions_reader,test_interaction_reader = read_files()
    interaction_matrix_csr =something(users_rows,items_rows,train_interactions_reader)
    if args.train == True:
        test_user_ids,index_to_selected_items  = train_data(users_rows,items_rows,train_interactions_reader)
    else:
        test_user_ids ,index_to_selected_items = test_data(train_interactions_reader,test_interaction_reader)
                                                              

    clust_to_items_csr,clust_to_users_csr = generate_data(users_rows,items_rows)
    N_WORKERS = args.n_workers

    bucket_size = len(test_user_ids) / N_WORKERS
    start = 0
    jobs = []
    for i in range(0, N_WORKERS):
        stop = int(min(len(test_user_ids), start + bucket_size))
        filename = args.save_filename + "_"+ str(i) + ".csv"
        print(start,stop)
        process = multiprocessing.Process(target = worker, args=( test_user_ids[start:stop],index_to_selected_items,clust_to_users_csr,clust_to_items_csr,interaction_matrix_csr, filename))
        jobs.append(process)
        start = stop

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

def something(users_rows,items_rows,interactions_reader):
    X_train, _ = train_test_split(interactions_reader, test_size=0.1, random_state=42)
    print("Generating userid to index list ...")
    user_counter = 0
    for row in users_rows:
        userid_to_index[row[1]] = user_counter
        users_matrix.append(row)
        user_counter += 1
        if user_counter == num_users:
            break

    print("Generating itemid to index list ...")
    item_counter = 0
    for row in items_rows:
        itemid_to_index[row[1]] = item_counter
        items_matrix.append(row)
        item_counter += 1
        if item_counter == num_items:
            break
    #index_to_items = {v: k for k, v in itemid_to_index.items()}
      

    print("Generating a sparse users X items matrix for interaction ...")
    num_lines_skipped = 0

    num_matches = 0
    for index, row in X_train.iterrows():
        if str(row['user_id']) in userid_to_index:
            num_matches += 1
            try:
                temp = float(row['interaction_type'])
                if temp == 4.0:
                    temp = -1.0
                interaction_matrix[userid_to_index[str(row['user_id'])], itemid_to_index[str(row['item_id'])]] = temp
            except:
                num_lines_skipped += 1

    print(num_matches)
    print(num_lines_skipped)
    interaction_matrix_csr = interaction_matrix.tocsr()

    return interaction_matrix_csr       


def train_data(users_rows,items_rows,interactions_reader): 
    
    
    _, X_test = train_test_split(interactions_reader, test_size=0.1, random_state=42)

    test_user_ids=[]
    print("Reading target_users.csv file ...")
    target_users_reader = pd.DataFrame(X_test['user_id'].drop_duplicates())
    for index,row in target_users_reader.iterrows():
        if str(row['user_id']) in userid_to_index:
            test_index = userid_to_index[str(row['user_id'])]
            test_user_ids.append(str(row['user_id']))
    biased_items_to_index={}
    target_items_reader = pd.DataFrame(X_test['item_id'].drop_duplicates())
    for ix,row in target_items_reader.iterrows():
        try: 
            biased_items_to_index[str(row['item_id'])] = itemid_to_index[str(row['item_id'])]
        except:
            pass     
    index_to_selected_items = {v: k for k, v in biased_items_to_index.items()}      

   

    return test_user_ids ,index_to_selected_items       


def test_data(train_interactions_reader,test_interaction_reader): 
    target_users_reader = pd.DataFrame(test_interaction_reader['user_id'].drop_duplicates())
    test_user_ids=[]
    unbiased_items_to_index={}
    for index,row in target_users_reader.iterrows():
        if str(row['user_id']) in userid_to_index:
            test_index = userid_to_index[str(row['user_id'])]
            test_user_ids.append(str(row['user_id']))
    target_items_reader = pd.DataFrame(test_interaction_reader['item_id'].drop_duplicates())
    for ix,row in target_items_reader.iterrows():
        try: 
            unbiased_items_to_index[str(row['item_id'])] = itemid_to_index[str(row['item_id'])]
        except:
            pass         

    index_to_selected_items = {v: k for k, v in unbiased_items_to_index.items()}      

    print("Generating a sparse users X items matrix for interaction ...")
    num_lines_skipped = 0



    return test_user_ids ,index_to_selected_items       

if __name__ == "__main__":
    main()


# -----------------------------------------------------------
