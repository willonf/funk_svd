def split_dataset(rating_matrix):
    users = rating_matrix.select('userId').sort('userId').unique().to_series().to_list()
    users_length = len(users)
    train_data_len = int(users_length * 70 / 100)
    test_users = set(random.sample(users, (users_length - train_data_len)))
    train_users = users - test_users

    return [], []