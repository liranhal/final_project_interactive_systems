import time

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.sparse import save_npz, load_npz, hstack, coo_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

def main():
    baseline_features_df = pd.read_csv('baseline_features.csv')
    baseline_features_cols = ['unigrams_perplexity', 'bigrams_perplexity', 'trigrams_perplexity',
                              'fourgrams_perplexity', 'flesch_readablity', 'automated_readability',
                              'positive_sent', 'neutral_sent', 'negative_sent', 'definite_vs_indefinite',
                              'context_sim', 'anomaly_sim', 'max_absolute_diff', 'average_absolute_diff']
    print(baseline_features_df.head())
    print(baseline_features_df.columns)

    all_contests = np.random.choice(list(set(baseline_features_df['contest'].values)), size=10, replace=False)
    print(all_contests)
    print('All contests length: ', len(all_contests))

    X_baseline_features = baseline_features_df[baseline_features_cols]
    X_baseline_features[X_baseline_features == float('inf')] = 250000
    print(baseline_features_df['score'])
    # y = baseline_features_df['score'].apply(lambda x: round(x))
    y = baseline_features_df['score'].apply(lambda x: x)


    bow_vectors = load_npz('bow_features.npz')

    print(X_baseline_features.shape)
    print(bow_vectors.shape)

    # stacked_data = hstack([X_baseline_features, bow_vectors])

    # train_stacked_data, test_stacked_data, y_train, y_test = train_test_split(stacked_data, y, test_size=0.33,
    #                                                                           random_state=42)

    train_contests, test_contests = train_test_split(all_contests, test_size=0.33, random_state=42)

    train_mask = baseline_features_df['contest'].apply(lambda x: x in train_contests).to_numpy()

    test_indexes_by_contest = {}
    for contest in test_contests:
        test_mask = baseline_features_df['contest'].apply(lambda x: x == contest).to_numpy()
        test_indexes = baseline_features_df.index[test_mask].tolist()
        test_indexes_by_contest[contest] = test_indexes

    train_indexes = baseline_features_df.index[train_mask].tolist()

    train_stacked_data, y_train = hstack([X_baseline_features.to_numpy()[train_indexes], bow_vectors[train_indexes]]),\
                                  y.to_numpy()[train_indexes]
    test_data_by_contest = {}
    for contest in test_contests:
        test_indexes = test_indexes_by_contest[contest]
        test_stacked_data, y_test = hstack([X_baseline_features.to_numpy()[test_indexes], bow_vectors[test_indexes]]),\
                                    y.to_numpy()[test_indexes]
        test_data_by_contest[contest] = (test_stacked_data, y_test)

    # print(train_stacked_data.shape, test_stacked_data.shape)
    # print(y_train)

    rfc = RandomForestRegressor()
    start = time.time()
    rfc.fit(train_stacked_data, y_train)
    print(f'Training time: {start-time.time()}')
    map = 0
    for contest in test_contests:
        test_stacked_data, y_test = test_data_by_contest[contest]
        # y test is ordered
        prediction = rfc.predict(test_stacked_data)
        y_test[:10] = 1
        y_test[10:] = 0
        map += average_precision_score(y_test, prediction) / len(test_contests)
    print(f'Map score: {map}')





if __name__ == '__main__':
    main()