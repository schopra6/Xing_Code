#Metrics
import numpy as np
import pandas as pd
import math
def dcg(recommended_items,relevant_items):
    is_relevant = np.in1d(recommended_items[:30], relevant_items, assume_unique=True).astype(np.float32)[:30]
    pos = 1
    gain = 0
    for i in is_relevant:
        gain+= (math.pow(2,i)-1)/math.log(pos+1)
        pos+=1
    return gain

def ndcg(recommended_items,relevant_items):
    return dcg(recommended_items,relevant_items)/float(dcg(relevant_items,relevant_items))

def ap_at_k(recommended_items, relevant_items, k):
        is_relevant = np.in1d(recommended_items[:k], relevant_items, assume_unique=True).astype(np.float32)
        total_relevant = len(relevant_items)   
        score = np.sum(np.cumsum(is_relevant) * is_relevant / np.arange(1, k+1)) / min(k, total_relevant)
        #assert 0 <= score <= 1
        return score

def precision_at_k(recommended_items, relevant_items, k):
	return float(np.intersect1d(recommended_items[:k], relevant_items).shape[0]) / k

def recall(recommended_items, relevant_items):
    total_relevant = len(relevant_items)
    return float(np.intersect1d(recommended_items[:30], relevant_items).shape[0]) / min(30, total_relevant)

def user_success(recommended_items, relevant_items):
	return float(np.intersect1d(recommended_items[:30], relevant_items).shape[0] > 0)

def return_metrics(S, T, k=30):
    recall_avg,prec_avg, user_success_avg, map_at_2, map_at_5, map_at_10, map_at_20, avgndcg = [], [], [], [], [], [], [],[]
    count = 0   
    for uid in S.index:
      if uid in T.index:  
        #parse ids
        recommended_items = np.array(S.loc[uid].values[0])
        relevant_items = np.array(T.loc[uid].values[0])
        
        #compute metrics for a user
        prec = precision_at_k(recommended_items, relevant_items, 5)
        rec = recall(recommended_items, relevant_items)      
        usucc = user_success(recommended_items, relevant_items)
        normdcg = ndcg(recommended_items, relevant_items)

        #update averages
        prec_avg.append(prec)
        recall_avg.append(rec)
        user_success_avg.append(usucc)
        avgndcg.append(normdcg)
        
        map_at_2.append(ap_at_k(recommended_items, relevant_items, 2))
        map_at_5.append(ap_at_k(recommended_items, relevant_items, 5))
        map_at_10.append(ap_at_k(recommended_items, relevant_items, 10))
        map_at_20.append(ap_at_k(recommended_items, relevant_items, 20))
      else:
          count +=1    
        
    return (recall_avg, user_success_avg, map_at_2, map_at_5, map_at_10, map_at_20,prec_avg)


def compute_metrics(S, T, k=30):
    recall_avg, user_success_avg, map_at_2, map_at_5, map_at_10, map_at_20, avgndcg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    count = 0   
    for uid in S.index:
      if uid in T.index:  
        #parse ids
        recommended_items = np.array(S.loc[uid].values[0])
        relevant_items = np.array(T.loc[uid].values[0])
        
        #compute metrics for a user
        prec = precision_at_k(recommended_items, relevant_items, 5)
        rec = recall(recommended_items, relevant_items)      
        usucc = user_success(recommended_items, relevant_items)
        normdcg = ndcg(recommended_items, relevant_items)

        #update averages
        recall_avg += rec
        user_success_avg += usucc
        avgndcg += normdcg
        
        map_at_2 += ap_at_k(recommended_items, relevant_items, 2)
        map_at_5 += ap_at_k(recommended_items, relevant_items, 5)
        map_at_10 += ap_at_k(recommended_items, relevant_items, 10)
        map_at_20 += ap_at_k(recommended_items, relevant_items, 20)
      else:
          count +=1    
        
    recall_avg /= (S.shape[0] -count)
    user_success_avg /= (S.shape[0]-count)
    map_at_2 /= (S.shape[0] - count)
    map_at_5 /= (S.shape[0] - count)
    map_at_10 /= (S.shape[0] - count)
    map_at_20 /= (S.shape[0] - count)
    
    return (recall_avg, user_success_avg, map_at_2, map_at_5, map_at_10, map_at_20)

#In caso qualcuno fosse curioso...
def challenge_score(S, T, k=20):
    score = 0.0
    recall_avg, user_success_avg, map_at_20 = 0.0, 0.0, 0.0
    count =0
    for uid in S.index:
      if uid in T.index:

        recommended_items = np.array(S.loc[uid].values[0])
        relevant_items = np.array(T.loc[uid].values[0])

        rec = recall(recommended_items, relevant_items)
        usucc = user_success(recommended_items, relevant_items)
        recall_avg += rec
        user_success_avg += usucc
        map_at_20 += ap_at_k(recommended_items, relevant_items, k)
        score += \
        20 * (
            precision_at_k(recommended_items, relevant_items, 2) +
            precision_at_k(recommended_items, relevant_items, 4) +
            usucc +
            rec
            ) + \
        10 * (
            precision_at_k(recommended_items, relevant_items, 6) +
            precision_at_k(recommended_items, relevant_items, 20)
            )
      else:
          count +=1 
    recall_avg /= (S.shape[0] - count)
    user_success_avg /= (S.shape[0] -count)
    map_at_20 /= (S.shape[0] - count)
    return score

def get_p(attribute,count): 
   p={attrib:count[attrib]/sum(count.values()) for attrib in attribute }
   print(sum(count.values()))
   return p


def totalRelevant(S, T, k=30): 
    for uid in S.index:
      if uid in T.index:  
        #parse ids
        recommended_items = np.array(S.loc[uid].values[0])
        relevant_items = np.array(T.loc[uid].values[0]) 
        is_relevant = np.in1d(recommended_items[:k], relevant_items, assume_unique=True).astype(np.float32)
        return is_relevant


def totalRelevantcount(S, T):
    '''
    Input:
     S : Recommended Items for a set of users
     R :  Relevant Items(Ground Truth) for a set of users
    '''
    relevant_count=0
    for uid in S.index:
      if uid in T.index:  
        #parse ids
        recommended_items = np.array(S.loc[uid].values[0])
        relevant_items = np.array(T.loc[uid].values[0])
        relevant_count += len(np.intersect1d(recommended_items, relevant_items, assume_unique=False))   
    return relevant_count

def GCE(attribute,p,pf,alpha=-1):
    sum=0
    for attrib in attribute:
        sum+=np.power(pf[attrib],alpha)*np.power(p[attrib],1-alpha) 
    sum = sum -1
    return abs(sum/(alpha*(1-alpha))) 