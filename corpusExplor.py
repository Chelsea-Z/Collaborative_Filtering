
#%%
u = df.loc[df['UserID']==4321]
u.groupby('Rating')['MovieID'].count()
u['Rating'].mean()

#%%
m = df.loc[df['MovieID']==3]
m.groupby('Rating')['UserID'].count()
m['Rating'].mean()

#%%
#find knn for 4321
k=5
for query in [4321]:

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

#%% 
#find knn for movie 3
k=5
A_dot = movie_user.dot(movie_user.T)
m_norm = LA.norm(movie_user, axis = 1) #norm of every movie
A_cosine = A_dot.divide(np.outer(m_norm,m_norm))

m = 3
#find knn with dot similarity
sorted_dot = A_dot[m].sort_values(ascending = False)
k_dot = sorted_dot.iloc[1:k+1].keys() #knn movies

#find knn with cosine similarity
sorted_cosine = A_cosine[m].sort_values(ascending = False)
k_cosine = sorted_cosine.iloc[1:k+1].keys()