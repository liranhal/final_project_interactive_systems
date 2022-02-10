from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import save_npz, load_npz, hstack
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    baseline_features_df = pd.read_csv('baseline_features.csv')
    baseline_features_cols = ['unigrams_perplexity', 'bigrams_perplexity', 'trigrams_perplexity',
                              'fourgrams_perplexity', 'flesch_readablity', 'automated_readability',
                              'positive_sent', 'neutral_sent', 'negative_sent', 'definite_vs_indefinite',
                              'context_sim', 'anomaly_sim', 'max_absolute_diff', 'average_absolute_diff']
    X_baseline_features = baseline_features_df[baseline_features_cols]
    y = baseline_features_df['score'].apply(lambda x: round(x))

    bow_vectors = load_npz('bow_features.npz')

    stacked_data = hstack(X_baseline_features, bow_vectors)

    train_stacked_data, test_stacked_data, y_train, y_test = train_test_split(stacked_data, y, test_size=0.33,
                                                                              random_state=42)

    rfc = RandomForestClassifier()
    rfc.fit(train_stacked_data, y_train)

    print(rfc.score(test_stacked_data, y_test))





if __name__ == '__main__':
    main()