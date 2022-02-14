from bert_classifier import BertHumorNetwork
import bert_dataset
import torch
from torch.nn import MSELoss
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score

def main():
    epochs = 100
    train_dataset_path = ''
    test_dataset_path = ''

    train_dataset = bert_dataset.BertDataset(train_dataset_path)
    test_dataset = bert_dataset.BertDataset(test_dataset_path)

    train_dataloader = torch.utils.data.DataLoader(train_dataset)
    test_dataloader = torch.utils.data.DataLoader(train_dataset)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertHumorNetwork()  # You can pass the parameters if required to have more flexible model
    model.to(torch.device("0"))  ## can be gpu
    criterion = MSELoss()  ## If required define your own criterion
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()))


    for epoch in epochs:
        for batch in train_dataloader:  ## If you have a DataLoader()  object to get the data.

            data = batch[0]
            targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets

            optimizer.zero_grad()
            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                                   add_special_tokens=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_true_label_list = []
        train_pred_list = []
        for batch in train_dataloader:  ## If you have a DataLoader()  object to get the data.

            data = batch[0]
            targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets
            top_ten_rank = batch[2]

            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                                   add_special_tokens=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)

            train_true_label_list.append(top_ten_rank)
            train_pred_list.append(outputs)

        test_indexes_by_contest = {}
        for contest in test_contests:
            test_mask = baseline_features_df['contest'].apply(lambda x: x == contest).to_numpy()
            test_indexes = baseline_features_df.index[test_mask].tolist()
            test_indexes_by_contest[contest] = test_indexes

        train_indexes = baseline_features_df.index[train_mask].tolist()

        test_data_by_contest = {}
        for contest in test_contests:
            test_indexes = test_indexes_by_contest[contest]
            test_stacked_data, y_test = hstack(
                [X_baseline_features.to_numpy()[test_indexes], bow_vectors[test_indexes]]), \
                                        y.to_numpy()[test_indexes]
            test_data_by_contest[contest] = (test_stacked_data, y_test)

        map = 0
        for contest in test_contests:
            test_stacked_data, y_test = test_data_by_contest[contest]
            # y test is ordered
            prediction = rfc.predict(test_stacked_data)
            y_test[:10] = 1
            y_test[10:] = 0
            map += average_precision_score(y_test, prediction) / len(test_contests)

        for batch in test_dataloader:  ## If you have a DataLoader()  object to get the data.

            data = batch[0]
            targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets
            top_ten_rank = batch[2]

            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                                   add_special_tokens=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
