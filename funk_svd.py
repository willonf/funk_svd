import math
import time

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


class FunkSVD:
    def __init__(self, k=40, learning_rate=0.001, regulation=0.02, iterations=10):
        self.k = k
        self.learning_rate = learning_rate
        self.regulation = regulation
        self.iterations = iterations

        self.global_mean = None
        self.users_bias = None
        self.items_bias = None
        self.P_matrix = None
        self.Q_matrix = None
        self.rmse_train_errors = None
        self.rmse_test_error = None

    def fit(self, X_user_item, y_rating):
        rating_n_rows = y_rating.shape[0]

        self.global_mean = y_rating.select('rating').mean()['rating'][0]

        n_users = X_user_item.select('userId').max().rows()[0][0]
        n_items = X_user_item.select('movieId').max().rows()[0][0]

        self.users_bias = np.zeros(shape=n_users)
        self.items_bias = np.zeros(shape=n_items)

        self.P_matrix = np.full(shape=(n_users, self.k), fill_value=0.1, dtype=np.float64)
        self.Q_matrix = np.full(shape=(n_items, self.k), fill_value=0.1, dtype=np.float64)

        _rmse_train_errors = np.zeros(shape=self.iterations)

        for n_iteration in range(self.iterations):
            print(f'TRAINING - ITERAÇÃO: {n_iteration}')
            sq_error = 0
            for r_matrix_row in range(rating_n_rows):

                user = X_user_item[r_matrix_row].select('userId').rows()[0][0] - 1
                item = X_user_item[r_matrix_row].select('movieId').rows()[0][0] - 1
                real_r = y_rating[r_matrix_row].select('rating').rows()[0][0]

                pred_r = self.global_mean + self.users_bias[user] + self.items_bias[item] + np.dot(self.P_matrix[user], self.Q_matrix[item])
                error_ui = real_r - pred_r
                sq_error = sq_error + error_ui ** 2

                self.users_bias[user] = self.users_bias[user] + self.learning_rate * (error_ui - self.regulation * self.users_bias[user])
                self.items_bias[item] = self.items_bias[item] + self.learning_rate * (error_ui - self.regulation * self.items_bias[item])

                for factor in range(self.k):
                    temp_uf = self.P_matrix[user, factor]
                    self.P_matrix[user, factor] = self.P_matrix[user, factor] + self.learning_rate * (error_ui * self.Q_matrix[item, factor] - self.regulation * self.P_matrix[user, factor])
                    self.Q_matrix[item, factor] = self.Q_matrix[item, factor] + self.learning_rate * (error_ui * temp_uf - self.regulation * self.Q_matrix[item, factor])
            _rmse_train_errors[n_iteration] = math.sqrt(sq_error / rating_n_rows)  # RMSE
            print(f'TRAINING - RMSE: {math.sqrt(sq_error / rating_n_rows)}')

        self.rmse_train_errors = pl.DataFrame(np.array([_rmse_train_errors, list(range(1, self.iterations + 1))]), schema=[("error", pl.Float64), ("iteration", pl.Int64)], orient="col")
        self.rmse_train_errors.write_csv('./_rmse_train_errors.csv')

    def predict_rating(self, user_id, item_id):
        return round(self.global_mean + self.users_bias[user_id - 1] + self.items_bias[item_id - 1] + np.dot(self.P_matrix[user_id - 1], self.Q_matrix[item_id - 1]), 2)

    def evaluate(self, X_user_item, y_rating):
        rating_n_rows = X_user_item.shape[0]
        sq_error = 0

        for row_index in range(rating_n_rows):
            user_id = X_user_item[row_index].rows()[0][0]
            item_id = X_user_item[row_index].rows()[0][1]
            real_rating = y_rating[row_index].rows()[0][0]
            predicted_rating = self.predict_rating(user_id, item_id)

            error_ui = real_rating - predicted_rating
            sq_error = sq_error + error_ui ** 2
        self.rmse_test_error = math.sqrt(sq_error / rating_n_rows)


if __name__ == '__main__':
    df_movies = pl.read_csv('./movies.csv')
    df_ratings = pl.read_csv('./ratings.csv')
    df_ratings = df_ratings.drop('timestamp')
    df_ratings = df_ratings.sort('userId')
    # df_ratings = df_ratings[:100]  # TODO: REMOVER!

    X = df_ratings.select(['userId', 'movieId'])
    y = df_ratings.select('rating')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = FunkSVD()

    print(f'INICIANDO TREINAMENTO...')
    start_trainning = time.time()
    model.fit(X_train, y_train)
    print(f'FINALIZANDO TREINAMENTO...')
    end_trainning = time.time()

    print(f'INICIANDO VALIDAÇÃO...')
    start_evaluation = time.time()
    model.evaluate(X_test, y_test)
    print(f'FINALIZANDO VALIDAÇÃO...')
    end_evaluation = time.time()

    print(model.rmse_test_error)
