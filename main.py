import pandas as pd 
import numpy as np 
from numpy import linalg as LA
import time 
import math 

def uuSimilarity(k, movie_user, dev_user):
### exp 1: User User similarity
    start = time.time()

    #k = 10
    dot_mean = {}
    cosine_mean = {}
    cosine_wgt = {}

    for query in dev_user:
    #for query in [0]:
        if query in movie_user.columns:
            q_vector = movie_user[query]

            #calculate cosine similarity and dot similarity
            dot = movie_user.T.dot(q_vector)
            q_norm = LA.norm(q_vector)
            u_norm = LA.norm(movie_user, axis = 0)
            cosine = dot.divide(q_norm*u_norm)

            # find knn with dot similarity
            sorted_dot = dot.sort_values(ascending= False)
            topk_dot = sorted_dot.iloc[1:k+1].keys() #knn without itself
            X_dot = movie_user[topk_dot]
        
            #find knn with cosine similarity
            sorted_cosine = cosine.sort_values(ascending = False)
            topk_cosine = sorted_cosine.iloc[1:k+1].keys()
            X_cosine = movie_user[topk_cosine]

            #predict dot mean
            pred_mean = X_dot.mean(axis = 1)+3
            dot_mean[query] = pred_mean

            #predict cosine mean
            cs_mean = X_cosine.mean(axis = 1)+3
            cosine_mean[query] = cs_mean.fillna(3)

            #predict cosine weighted mean
            weight = cosine[topk_cosine]
            weight = weight/np.sum(weight)
            index = cs_mean.index
            cs_wgt = pd.Series(np.average(X_cosine, axis=1, weights=weight)+3, index = index)
            cosine_wgt[query] = cs_wgt.fillna(3)
        else:
            X_dot = movie_user
            X_cosine = movie_user

            #predict dot mean
            pred_mean = X_dot.mean(axis = 1)+3
            dot_mean[query] = pred_mean

            #predict cosine mean
            cs_mean = X_cosine.mean(axis = 1)+3
            cosine_mean[query] = cs_mean

            #predict cosine weighted mean
            cs_wgt = X_cosine.mean(axis = 1)+3
            cosine_wgt[query] = cs_wgt

    dot_mean_list = []
    cosine_mean_list = []
    cosine_wgt_list = []
    for index, row in dev_user.iterrows():
        m = row['MovieID']
        u = row['UserID']

        if m in dev_user.index:
            dot_mean_list.append(dot_mean[u][m])
            cosine_mean_list.append(cosine_mean[u][m])
            cosine_wgt_list.append(cosine_wgt[u][m])
        else: 
            dot_mean_list.append(3)
            cosine_mean_list.append(3)
            cosine_wgt_list.append(3)

    with open('test_dot_mean.txt', 'w') as f:
        for rating in dot_mean_list:
            f.write("%s\n" % rating)

    with open('test_cosine_mean.txt', 'w') as f:
        for rating in cosine_mean_list:
            f.write("%s\n" % rating)

    with open('test_cosine_wgt.txt', 'w') as f:
        for rating in cosine_wgt_list:
            f.write("%s\n" % rating)

    end = time.time()
    print('exp1: ', end-start)

def mmSimilarity(k, movie_user, dev_user, dev):
### Exp 2: Movie Movie Similarity
    start = time.time()

    #k = 10
    dot_mean = []
    cosine_mean = []
    cosine_wgt = []

    #compute dot movie-movie dot similarity & cosine similarity
    A_dot = movie_user.dot(movie_user.T)
    m_norm = LA.norm(movie_user, axis = 1) #norm of every movie
    A_cosine = A_dot.divide(np.outer(m_norm,m_norm))

    for index, row in dev.iterrows():
        m = row['MovieID']
        u = row['UserID']
        if (m in A_dot.columns) and (u in movie_user.columns):
            #find knn with dot similarity
            sorted_dot = A_dot[m].sort_values(ascending = False)
            k_dot = sorted_dot.iloc[1:k+1].keys() #knn movies

            #find knn with cosine similarity
            sorted_cosine = A_cosine[m].sort_values(ascending = False)
            k_cosine = sorted_cosine.iloc[1:k+1].keys()

            u_vector = movie_user[u]

            #predict dot mean
            dt_mean = u_vector[k_dot].mean()+3
            dot_mean.append(dt_mean)

            #predict cosine mean
            cs_mean = u_vector[k_cosine].mean()+3
            cosine_mean.append(cs_mean)
            #predict cosine wight mean
            weight = A_cosine[m][k_cosine]
            weight = weight/np.sum(weight)
            cs_wgt = np.average(u_vector[k_cosine], weights = weight)+3
            if math.isnan(cs_wgt):
                cosine_wgt.append(3)
            else:
                cosine_wgt.append(cs_wgt)
            
        else:
            dot_mean.append(3)
            cosine_mean.append(3)
            cosine_wgt.append(3)

    with open('exp2_dot_mean.txt', 'w') as f:
        for rating in dot_mean:
            f.write("%s\n" % rating)

    with open('exp2_cosine_mean.txt', 'w') as f:
        for rating in cosine_mean:
            f.write("%s\n" % rating)

    with open('exp2_cosine_wgt.txt', 'w') as f:
        for rating in cosine_wgt:
            f.write("%s\n" % rating)

    end = time.time()
    print('exp2: ', end-start)

def pcc(k, movie_user, dev_user, dev):
### Exp 3: Normalization
    start = time.time()

    dot_mean = []
    cosine_mean = []
    cosine_wgt = []

    #normalization on movie
    mov_mean = movie_user.mean(axis = 1)
    mov_std = movie_user.std(axis = 1)
    X_norm = movie_user.subtract(mov_mean, axis =0).divide(mov_std, axis = 0).fillna(0)

    #compute dot movie-movie dot similarity & cosine similarity
    A_dot = X_norm.dot(X_norm.T)
    m_norm = LA.norm(X_norm, axis = 1) #norm of every movie
    A_cosine = A_dot.divide(np.outer(m_norm,m_norm))

    for index, row in dev.iterrows():
        m = row['MovieID']
        u = row['UserID']
        if (m in A_dot.columns) and (u in X_norm.columns):
            #find knn with dot similarity
            sorted_dot = A_dot[m].sort_values(ascending = False)
            k_dot = sorted_dot.iloc[1:k+1].keys() #knn movies

            #find knn with cosine similarity
            sorted_cosine = A_cosine[m].sort_values(ascending = False)
            k_cosine = sorted_cosine.iloc[1:k+1].keys()

            u_vector = movie_user[u]

            #predict dot mean
            dt_mean = u_vector[k_dot].mean()+3
            dot_mean.append(dt_mean)

            #predict cosine mean
            cs_mean = u_vector[k_cosine].mean()+3
            cosine_mean.append(cs_mean)
            #predict cosine wight mean
            weight = A_cosine[m][k_cosine]
            weight = weight/np.sum(weight)
            cs_wgt = np.average(u_vector[k_cosine], weights = weight)+3
            if math.isnan(cs_wgt):
                cosine_wgt.append(3)
            else:
                cosine_wgt.append(cs_wgt)
            
        else:
            dot_mean.append(3)
            cosine_mean.append(3)
            cosine_wgt.append(3)

    with open('exp3_dot_mean.txt', 'w') as f:
        for rating in dot_mean:
            f.write("%s\n" % rating)

    with open('exp3_cosine_mean.txt', 'w') as f:
        for rating in cosine_mean:
            f.write("%s\n" % rating)

    with open('exp3_cosine_wgt.txt', 'w') as f:
        for rating in cosine_wgt:
            f.write("%s\n" % rating)

    end = time.time()
    print('exp3: ', end-start)

def main(k):

    #read train
    train = './HW3_data/train.csv'
    columns  = ['MovieID', 'UserID', 'Rating', 'RatingDate']
    df = pd.read_csv(train, header=None, names= columns)
    df.drop(columns = 'RatingDate', inplace=True)
    movie_user = df.pivot(index= 'MovieID', columns='UserID', values = 'Rating')
    movie_user.fillna(3, inplace=True)
    movie_user = movie_user.subtract(3) #imputation
    movie_user.head()

    #read dev
    filename = './HW3_data/dev.csv' #param
    columns = ['MovieID', 'UserID']
    dev = pd.read_csv(filename, header = None, names = columns)
    dev_user = list(set(dev['UserID']))

    uuSimilarity(k, movie_user, dev_user)
    mmSimilarity(k, movie_user, dev_user, dev)
    pcc(k, movie_user, dev_user, dev)

if __name__ == "__main__":
    main(10)