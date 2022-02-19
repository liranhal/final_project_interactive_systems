import torch
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.model_selection import train_test_split
import ast

def replace_list(x):
    x = str(x).replace(',', '')
    x = x.replace('[', '')
    x = x.replace(']', '')
    x = x.replace('\'', '')
    return x.replace('nan', '')


def create_bert_train_test_datasets(data_path, train_target_path, test_target_path, contests=None):
    data_path = data_path

    dataset = pd.read_csv(data_path)[['contest', 'caption', 'score', 'context_words_list', 'anomaly_words_list']]

    dataset = dataset[dataset['contest'].isin(contests)] # .reset_index()

    dataset['top_ten_rank'] = 0

    contests_count = {}
    # for contest in contests:
    #     ten_values = dataset[dataset['contest']][]
    top_ten_rank = list(dataset["top_ten_rank"])
    for i in range(len(dataset)):
        if dataset['contest'].iloc[i] in contests_count:
            if contests_count[dataset.iloc[i]['contest']] < 10:
                contests_count[dataset.iloc[i]['contest']] += 1
                # dataset.iloc[i]['top_ten_rank'] = 1
                top_ten_rank[i] = 1
                x = 5
                y = 3
        else:
            contests_count[dataset.iloc[i]['contest']] = 1
            # dataset.iloc[i]['top_ten_rank'] = 1
            top_ten_rank[i] = 1
    dataset["top_ten_rank"] = top_ten_rank
    x = 5

        # self.__snow_stemmer = SnowballStemmer(language='english')

    dataset['caption'] = dataset['caption'].apply(lambda x: preprocess(x))

    dataset['caption'] = dataset['caption'] + ' ' + dataset['context_words_list'].apply(lambda x: replace_list(x))\
                          + ' ' + dataset['anomaly_words_list'].apply(lambda x: replace_list(x))

    train_contests, test_contests = train_test_split(contests, test_size=0.33)

    train_dataset = dataset[dataset['contest'].isin(train_contests)]
    test_dataset = dataset[dataset['contest'].isin(test_contests)]

    train_dataset.to_csv(train_target_path, index=False)
    test_dataset.to_csv(test_target_path, index=False)


def preprocess(x):
    x = str(x).lower()
    return re.sub(r'[^\w\s]', '', x)
    # x_words = x.split()
    # x_stemmed_words = [self.__snow_stemmer.stem(str(w)) for w in x_words]
    # return ' '.join(x_stemmed_words)


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.__data_path = data_path

        self.__dataset = pd.read_csv(self.__data_path)

        self.contests = list(set(self.__dataset['contest'].values))

    def __getitem__(self, idx):
        return self.__dataset.iloc[idx]['caption'], self.__dataset.iloc[idx]['score'], \
               self.__dataset.iloc[idx]['top_ten_rank'], self.__dataset.iloc[idx]['contest']

    def __len__(self):
        return len(self.__dataset)


if __name__ == '__main__':
    create_bert_train_test_datasets('baseline_features.csv', 'train_bert_data.csv', 'test_bert_data.csv',
                                    contests=[580, 581, 582, 583, 584, 585, 586, 587, 588, 589])