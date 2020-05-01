import argparse
import numpy as np
import os
import pandas as pd
import surprise


from sklearn.decomposition import NMF, TruncatedSVD
from surprise import accuracy, SVD
from surprise import Reader


def preprocess_data(X_train, X_test):
    X_train = pd.concat([X_train, X_test])
    X_train.loc[X_test.index, "rating"] = np.NaN
    return X_train, X_test


def get_rse(prediction: pd.DataFrame, test_set: pd.DataFrame):
    ratings_comparison = prediction.merge(test_set, on=["userId", "movieId"]).loc[:, ["rating_x", "rating_y"]]
    return np.sqrt(np.mean((ratings_comparison["rating_x"] - ratings_comparison["rating_y"]) ** 2))


def create_util_matrix(X_train: pd.DataFrame):
    return X_train.pivot(index='userId', columns='movieId')


def adjust_to_mean(df: pd.DataFrame, util_matrix: pd.DataFrame, col: str):
    group_means = df.groupby(col).aggregate(np.mean)
    util_copy = util_matrix.copy()
    for col in util_copy.columns:
        util_copy.loc[:, col] = group_means.values.reshape(-1)
    return util_copy


def populate_util_matrix(X_train: pd.DataFrame, util_matrix: pd.DataFrame):
    grand_mean = X_train.loc[:, "rating"].mean()
    user_avg_ratings = adjust_to_mean(X_train.drop('movieId', axis=1), util_matrix, 'userId')
    movie_avg_ratings = adjust_to_mean(X_train.drop('userId', axis=1), util_matrix.T, 'movieId').T.fillna(grand_mean)
    mean_adjusted_ratings = user_avg_ratings + movie_avg_ratings - grand_mean
    return util_matrix.mask(np.isnan, other=mean_adjusted_ratings)


def truncated_svd(util_matrix, X_test):
    svd = TruncatedSVD(n_components=12, random_state=42)
    svd.fit(util_matrix)
    sigma = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(util_matrix) / svd.singular_values_
    H = np.dot(sigma, VT)
    svd_prediction = W @ H
    svd_prediction = pd.DataFrame(svd_prediction,
                                  columns=util_matrix.columns,
                                  index=util_matrix.index)
    return get_rse(svd_prediction.stack().reset_index(), X_test)


def iterative_svd(util_matrix, X_test):
    svd_iter = TruncatedSVD(n_components=5, random_state=1234)

    def if_converged():
        return np.all(np.isclose(Z_0, Z_next, rtol=1, atol=1))

    def if_exceed_max_iter(i, max_iter=35):
        return i > max_iter

    i = 1
    Z_0 = util_matrix.copy()
    nan_indices = util_matrix.isna()
    while not if_exceed_max_iter(i):
        svd_iter.fit(Z_0)
        sigma = np.diag(svd_iter.singular_values_)
        VT = svd_iter.components_
        W = svd_iter.transform(util_matrix) / svd_iter.singular_values_
        H = np.dot(sigma, VT)
        Z_next = W @ H
        Z_next = pd.DataFrame(Z_next, columns=util_matrix.columns, index=util_matrix.index)
        if if_converged():
            print('Converged')
            break
        Z_0 = Z_0.mask(nan_indices, Z_next)
        i += 1

    return get_rse(Z_0.stack(), X_test)


def nmf(util_matrix, X_test):
    epsilon = 10e-4
    nmf_offset = abs(np.min(util_matrix.values)) + epsilon
    nmf_model = NMF(n_components=19, random_state=1234)
    W = nmf_model.fit_transform(util_matrix + nmf_offset)
    H = nmf_model.components_
    X_approx = np.dot(W, H) - nmf_offset
    X_approx = pd.DataFrame(X_approx, columns=util_matrix.columns, index=util_matrix.index)
    return get_rse(X_approx.stack().reset_index(), X_test)


DECOMPOSITION_ALGO = {'SVD1': truncated_svd, 'SVD2': iterative_svd, 'NMF': nmf}


def run_decomposition(algo: str, X_train, X_test):
    util_matrix = create_util_matrix(X_train)
    util_matrix = populate_util_matrix(X_train, util_matrix)
    return DECOMPOSITION_ALGO[algo](util_matrix, X_test)


def svd_sgd(X_train, X_test):
    sgd_svd = SVD(**{'n_epochs': 30, 'reg_all': 0.05, 'lr_all': 0.009})
    sgd_svd.fit(X_train)
    predictions = sgd_svd.test(X_test)
    return accuracy.rmse(predictions)


def read_csv(path):
    return pd.read_csv(path).drop(["timestamp"], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommendation System")
    parser.add_argument('--train', default="train_ratings.csv")
    parser.add_argument('--test', default="test_ratings.csv")
    parser.add_argument('--alg', default="SGD", required=False)
    parser.add_argument('--result',
                        default=os.path.join(os.getcwd(), "wynik.txt"),
                        required=False,
                        help='result file  (default: %(default)s)')

    args = parser.parse_args()
    train_path, test_path, alg, result = args.train, args.test, args.alg, args.result
    X_train, X_test = read_csv(train_path), read_csv(test_path)

    # Testing purpose
    # alg, result = args.alg, args.result
    # ratings_path = os.path.join(os.getcwd(), os.path.join('ml-latest-small', 'ratings.csv'))
    # data = pd.read_csv(ratings_path).drop(["timestamp"], axis=1)
    # X_train, X_test = train_test_split(data, test_size=0.1, random_state=42)

    if alg == 'SGD':
        reader = Reader()
        X_train = surprise.Dataset.load_from_df(X_train, reader).build_full_trainset()
        X_test = surprise.Dataset.load_from_df(X_test, reader).build_full_trainset().build_testset()
        rse = svd_sgd(X_train, X_test)
    else:
        X_train, X_test = preprocess_data(X_train.sort_index(), X_test.sort_index())
        rse = run_decomposition(alg, X_train, X_test)

    print(rse)
