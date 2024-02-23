
import sys
sys.path.append("../")

import pandas as pd

from utils.common.timer import Timer
from Datasets import Movielens
from Evaluation.data_split import split_data
from Evaluation.ranking_metrics import *
from WRMF.wrmf import *
from WRMF import wrmf_rec
import tensorflow.compat.v1 as tf
from scipy.sparse import csr_matrix, vstack,hstack


# rating_path = './../split/train.csv' #'./../split/train.csv'#'./../metadata/rating_matrix.csv'
# save_lightfm_path = './lightfm.pkl'
# save_U_path = './U.csv.bin'
# save_V_path = './V.csv.bin'
# save_U_bias_path = './U_bias.csv.bin'
# save_V_bias_path = './V_bias.csv.bin'

parser  = argparse.ArgumentParser()

parser.add_argument('--rating_path',help='path for rating matrix',dest='rating_path',type=str)
parser.add_argument('--save_lightfm_path',help='path for storing lightfm model class',dest='save_lightfm_path',type=str)
parser.add_argument('--save_U_path',help='path for storing U matrix',dest='save_U_path',type=str)
parser.add_argument('--save_V_path',help='path for storing V matrix',dest='save_V_path',type=str)

args    = vars(parser.parse_args())

"""
USAGE:
python decompose.py --rating_path=./../split/train.csv --save_lightfm_path=./lightfm.pkl --save_U_path=./U.csv.bin
--save_V_path=./V.csv.bin --save_U_bias_path=./U_bias.csv.bin --save_V_bias_path=./V_bias.csv.bin
"""


factors = 200

def main(rating_path,save_lightfm_path,save_U_path,save_V_path):
    train = pd.read_csv(rating_path)
    models = [
    WRMF(train, weight_strategy="user_oriented", alpha=0.007, lambda_u=0.1,
         lambda_v=0.1, k=factors, learning_rate=0.01)]

    strategies = [
    "user_oriented"]
    mf
    cols = ["Data", "Strategy", "K", "Train time (s)","Precision@k", "Recall@k"]
    df_result = pd.DataFrame(columns=cols)
    k = 10
    for strategy, model in zip(strategies, models):
        # 1. train
        with Timer() as train_time:
            model =  train_cornac(model, train)
    V_new = np.zeros((models[0].V.shape[0]+1,200))
    V_new[1:,:] = models[0].V
    U_new = np.zeros((models[0].U.shape[0]+1,200))
    U_new[1:,:] = models[0].U
    train_data = csr_matrix((ratings, (row, col)), shape=(n_user, n_item))
    model = LightFM(loss='warp',no_components=200,item_alpha=0.001,user_alpha=0.001)
    model.fit(train_data, epochs=20, num_threads=30)

    print ("u_preference matrix shape: ", model.user_embeddings.shape)
    print ("v_preference matrix shape: ", model.item_embeddings.shape)


    U_new.astype('float32').tofile(open(save_U_path, 'w'))
    V_new.astype('float32').tofile(open(save_V_path, 'w'))



if __name__ == '__main__':
    main(**args)
    #test(train_matrix_path='./../../movielen1m/fake_data/train.csv',test_matrix_path='./../../movielen1m/fake_data/test_warm.csv')
    #test(train_matrix_path='./../split/train.csv',test_matrix_path='./../split/test_warm.csv')

    # rating_path = './../metadata/rating_matrix.csv'
    #
    # rating_datas  = pd.read_csv(rating_path,sep=',')
    # rating_matrix = rating_datas.as_matrix()
    #
    # U, V = matrix_decomposite(rating_matrix, k=200, n_iter=1000)
    #
    # U.astype('float32').tofile(open('U.csv.bin','w'))
    # V.astype('float32').tofile(open('V.csv.bin','w'))