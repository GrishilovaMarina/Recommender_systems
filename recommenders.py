import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

RANDOM_STATE = 42

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, top_popular_n, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data_in = data
        self.top_popular = top_popular_n

        self.user_item_matrix = self.prepare_matrix(data, top_popular_n)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 

        self.user_item_sparse_matrix = csr_matrix(self.user_item_matrix).tocsr()
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data_in, top_popular_n):
        
        # your_code

        data_in.loc[~data_in['item_id'].isin(top_popular_n), 'item_id'] = 999999

        user_item_matrix = pd.pivot_table(data_in, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробоват ьдругие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        user_item_sparse_matrix = csr_matrix(user_item_matrix).tocsr()

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(user_item_sparse_matrix)
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        user_item_sparse_matrix = csr_matrix(user_item_matrix).tocsr()

        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(user_item_sparse_matrix)
        
        return model

    def extend_from_top_popular(self, recommendations, N=5):
        """Если количество рекомендаций меньше N, то дополняем их топ-популярными"""
        
        max_top_popular_len = len(self.top_popular)
        recommendations = list(recommendations)
        if len(recommendations) < N:
            if N <= max_top_popular_len:
                top_popular = [rec for rec in self.top_popular[:N] if rec not in recommendations]
                recommendations.extend(top_popular)
                recommendations = recommendations[:N]
            else:
                recommendations = recommendations[:max_top_popular_len]
        return recommendations            



    def get_recommendation_for_user(self, user, N):
        
        res = [self.id_to_itemid[rec] for rec in 
                self.model.recommend(userid=self.userid_to_id[user], 
                                     user_items=self.user_item_sparse_matrix[self.userid_to_id[user]],   # на вход user-item matrix
                                     N=N, 
                                     filter_already_liked_items=False, 
                                     filter_items=[self.itemid_to_id[999999]],
                                     # filter_items=None, 
                                     recalculate_user=True)[0]]
        return res 
    
    def get_model_recommendation(self, N=5):
        res_model_recommendation = self.data_in['user_id'].to_frame().drop_duplicates(ignore_index=True)
        
        res_model_recommendation['model_rec'] = res_model_recommendation['user_id']\
                                                .apply(lambda x: self.get_recommendation_for_user(x, N=N))
        
        return res_model_recommendation

    def get_similar_items(self, x):
        similar_item = self.model.similar_items(self.itemid_to_id[x], N=2)[0][1]
        res = self.id_to_itemid[similar_item]
        return res

    def get_similar_items_recommendation(self, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popularity = self.data_in.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.groupby('user_id').head(N)
        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)

        popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: self.get_similar_items(x))

        recommendation_similar_items = popularity.groupby('user_id')['similar_recommendation'].unique().reset_index()
        recommendation_similar_items.columns=['user_id', 'similar_recommendation']

        recommendation_similar_items['similar_recommendation'] = \
            recommendation_similar_items['similar_recommendation'].apply(lambda x: self.extend_from_top_popular(x, N=N))

        return recommendation_similar_items

    def get_similar_users(self, user, N):
        similar_users = self.model.similar_users(self.userid_to_id[user], N=(N+1))[0]
        similar_users_id = [self.id_to_userid[user] for user in similar_users]
        return similar_users_id[1:]

    def get_similar_users_recommendation(self, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popularity = self.data_in.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.groupby('user_id').head(1)
        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)

        popularity['similar_users_items'] = \
            popularity['user_id']\
                .apply(lambda x: popularity[popularity['user_id'].isin(self.get_similar_users(x, N=N))].item_id.to_list())

        recommendation_similar_user_items = popularity[['user_id', 'similar_users_items']]

        return recommendation_similar_user_items

if __name__ == '__main__':
    pass



        

