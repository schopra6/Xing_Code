from sklearn.datasets import dump_svmlight_file
from scipy.sparse import csr_matrix,hstack
import pandas as pd
import numpy as np
import argparse


parser  = argparse.ArgumentParser()

parser.add_argument('--save_U_path',help='path for storing U matrix',dest='save_U_path',type=str)
parser.add_argument('--save_V_path',help='path for storing V matrix',dest='save_V_path',type=str)

args    = vars(parser.parse_args())


def map_cat_to_index(X,column):
	names = set(X[column])
	index = range(len(set(X[column])))
	cat_to_index = dict(zip(names,index))
	return np.array([cat_to_index[i] for i in X[column]]) 

def map_special_column(users,items,column):
    jobrole_list = sum(users.jobrole_list, [])
    tag_list = sum(items.tag_list, [])
    title_list = sum(items.title_list, [])
    item_tags = set(jobrole_list + tag_list +title_list)
    index = range(len(set(X[column]))) 
    #cat_to_index = dict(zip(names,index))
	  
    #return np.array([cat_to_index[i] for i in X[column]]) 

def convert_to_one_hot(X,cat_columns):
	for column in cat_columns:

		X[column] = map_cat_to_index(X,column)

	prev_max = max(X[cat_columns[0]])+1
	for column in cat_columns[1:]:
		col_max = max(X[column])
		X[column] = np.array([val+prev_max for val in X[column]])
		prev_max += col_max
	return X

def categorical_df_to_csr(X, y, cat_columns, num_columns=None,array_columns=None):
	"""Converts a categorical DataFrame X to a sparse CSR matrix.
	
	This takes a pandas DataFrame with categorical features and converts category
	values to a sparse one-hot representation.
	
	Parameters
	----------
	X : pandas DataFrame, shape = [n_samples, n_features]
	    Training vectors, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like, shape = [n_samples]
	    Target values.

	cat_columns: array-like
		List of categorical columns

	num_columns: array-like, optional
		List of numerical columns

	Returns
	-------
	sparse_X : sparse CSR matrix
		Sparse vectorized version of categorical DataFrame
	"""
	X = convert_to_one_hot(X,cat_columns)
	cat_col_indexes = np.concatenate([X[i] for i in X[cat_columns]])
	if len(cat_columns)>1:
		cat_rows = np.concatenate([X.index for i in range(len(cat_columns))])
	else:
		cat_rows = X.index
	max_col_value = X.max()[cat_columns[-1]]
	data = np.ones(len(cat_col_indexes))
	if num_columns is None:
		sparse_X = csr_matrix((data,(cat_rows,cat_col_indexes)))
	else:
		num_col_indexes = np.array([max_col_value +1+j for j in range(len(num_columns)) for i in range(len(X.index)) ])
		if len(num_columns)>1:
			#num_rows = [X.index for i in range(len(num_columns))]
			num_rows = np.tile(np.array(X.index),len(num_columns))
			all_col_indexes = np.concatenate((cat_col_indexes, num_col_indexes))	
		else:
			num_rows = X.index
			print(cat_col_indexes)
			print(num_col_indexes)
			all_col_indexes = np.concatenate((cat_col_indexes, num_col_indexes))
		rows = np.concatenate((cat_rows,num_rows))
		num_data = np.concatenate([X[i] for i in X[num_columns]])
		data = np.concatenate((data, num_data))
		sparse_X = csr_matrix((data,(rows,all_col_indexes)))
		max_col_value = num_col_indexes[-1] +1
		if array_columns:
			for column in array_columns:
				# Determine the maximum number of columns in the list of lists
				list_column = X[column]
				max_columns = 1000#max(len(row) for row in list_column)
 				# Create a lil_matrix with the same shape as the CSR matrix
				lil = lil_matrix((len(list_column), max_columns), dtype=float)
				 # Fill the lil_matrix with values from the list of lists
				for i, row in enumerate(list_column):
					lil[i, row] = 1
				# Convert the lil_matrix to a CSR matrix
				matrix = csr_matrix(lil)
				sparse_X = hstack([sparse_X,matrix])
	#if array_columns not None:
        
        
	return sparse_X

def dump_categorical_df_to_svm_light(X, y, filename, cat_columns,
									 num_columns=None,array_columns=None, zero_based=True,
									 comment=None, query_id=None):
	"""Converts a categorical DataFrame X to a sparse CSR matrix,
	then writes to SVM-lite format.

	This takes a pandas DataFrame with categorical features, converts category
	values to a sparse one-hot representation, then writes the sparse matrix
	to SVM-light format.

	SVM-light is a text-based format, with one sample per line. It does
	not store zero valued features hence is suitable for sparse dataset.

	The first element of each line can be used to store a target variable
	to predict.

	Parameters
	----------
	X : pandas DataFrame, shape = [n_samples, n_features]
	    Training vectors, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like, shape = [n_samples]
	    Target values.

	filename : string or file-like in binary mode
	    If string, specifies the path that will contain the data.
	    If file-like, data will be written to f. f should be opened in binary
	    mode.

	cat_columns: array-like
		List of categorical columns

	num_columns: array-like, optional
		List of numerical columns

	zero_based : boolean, optional
	    Whether column indices should be written zero-based (True) or one-based
	    (False).

	comment : string, optional
	    Comment to insert at the top of the file. This should be either a
	    Unicode string, which will be encoded as UTF-8, or an ASCII byte
	    string.

	query_id : array-like, shape = [n_samples]
	    Array containing pairwise preference constraints (qid in svmlight
	    format).

    Examples
    --------
	import pandas as pd
	import numpy as np

	category_data_1 = ['tcp','udp','udp','tcp','dns','tcp']
	category_data_2 = ['red','blue','red','green','blue','red']
	numerical_data = [1,2,1,1,3,4]
	data  = {'category_data_1': category_data_1,
			 'category_data_2': category_data_2,
            'numerical_data_1':numerical_data,
			 'numerical_data_2':numerical_data}
	X = pd.DataFrame(data)
	y = np.array([1.,0.,1.,1.,0.,0.])
	
	cat_columns = ['category_data_1', 'category_data_2']
	num_columns = ['numerical_data_1','numerical_data_1']

	dump_categorical_df_to_svm_light(X, y, 'example', cat_columns, num_columns)

	head example	
	# Generated by dump_svmlight_file from scikit-learn 0.13-git
	# Column indices are zero-based
	1.000000 2:1.0000000000000000e+00 4:2.0000000000000000e+00
	0.000000 0:1.0000000000000000e+00 2:1.0000000000000000e+00 4:2.0000000000000000e+00
	1.000000 0:1.0000000000000000e+00 4:2.0000000000000000e+00
	1.000000 2:1.0000000000000000e+00 3:1.0000000000000000e+00 4:1.0000000000000000e+00
	0.000000 1:1.0000000000000000e+00 2:1.0000000000000000e+00 4:3.0000000000000000e+00
	0.000000 2:1.0000000000000000e+00 4:5.0000000000000000e+00
	"""
	if num_columns is None:
		sparse_X = categorical_df_to_csr(X, y, cat_columns)
	else:
		sparse_X = categorical_df_to_csr(X, y, cat_columns,num_columns,array_columns)
	dump_svmlight_file(sparse_X,y,filename)

def main(save_U_path,save_V_path):
    train = pd.read_csv(rating_path)
    users = pd.read_csv('assigned_users.csv')
    items['title']=items['title'].fillna('-1')
    users['jobroles_list'] = users['jobroles'].apply(lambda x: [int(ele ) for ele in str(x).split(",")])
    items['title_list'] = items['title'].apply(lambda x: [int(ele ) for ele in str(x).split(",")])
    items['tag_list'] = items['tags'].apply(lambda x: [int(ele ) for ele in str(x).split(",")])
    

    save_U_path ='user_content.dat'
    save_V_path = 'item_content.dat'
    cat_columns = ['country', 'region']
    num_columns = ['career_level', 'discipline_id',
               'experience_n_entries_class','experience_years_experience', 
               'experience_years_in_current',
               'edu_degree','gender']
    array_columns =['jobroles_list']


    dump_categorical_df_to_svm_light(users, users['user_id'], save_U_path, cat_columns, num_columns,array_columns)
    cat_columns = ['country', 'region']
    num_columns = ['career_level', 'discipline_id', 'industry_id','latitude', 'longitude','employment']
    array_columns =['title_list']
    dump_categorical_df_to_svm_light(items, items['id'], save_V_path, cat_columns, num_columns,array_columns)


if __name__ == '__main__':
    main(**args)   
