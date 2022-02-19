import pandas as pd

from bert_classifier import BertHumorNetwork
import bert_dataset
import torch
from torch.nn import MSELoss
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
from tqdm import tqdm


def main():
    epochs = 100
    batch_size = 64
    train_dataset_path = 'train_bert_data.csv'
    test_dataset_path = 'test_bert_data.csv'

    last = 'last_weight.pt'

    train_dataset = bert_dataset.BertDataset(train_dataset_path)
    test_dataset = bert_dataset.BertDataset(test_dataset_path)

    train_contests = train_dataset.contests
    test_contests = test_dataset.contests

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertHumorNetwork()  # You can pass the parameters if required to have more flexible model
    model.to(device)  ## can be gpu
    criterion = MSELoss()  ## If required define your own criterion
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    train_map_lst = []
    test_map_lst = []

    for epoch in range(epochs):
        for batch in train_dataloader:  ## If you have a DataLoader()  object to get the data.
            # print(batch)
            data = list(batch[0])
            targets = batch[1].to(device).float() ## assuming that data loader returns a tuple of data and its targets

            optimizer.zero_grad()
            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                                   add_special_tokens=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            # print(input_ids)
            # print(attention_mask)
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print("evaluating mAP for train set")

        train_true_label_list = []
        train_pred_list = []
        train_pred_contests = []
        for batch in train_dataloader:   # If you have a DataLoader()  object to get the data.

            data = list(batch[0])
            targets = batch[1]   # assuming that data loader returns a tuple of data and its targets
            top_ten_rank = batch[2].tolist()
            b_contests = batch[3].tolist()

            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=50, add_special_tokens=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask).tolist()

            train_true_label_list += top_ten_rank
            train_pred_list += outputs
            train_pred_contests += b_contests
        contests_scores_df = pd.DataFrame(data={'contest': train_pred_contests, 'train_pred_list': train_pred_list,
                                                'train_true_label_list': train_true_label_list})

        train_map = 0
        for contest in train_contests:
            cur_contest_data = contests_scores_df[contests_scores_df['contest'] == contest]
            # y test is ordered
            y_test = cur_contest_data['train_true_label_list'].tolist()
            prediction = cur_contest_data['train_pred_list'].tolist()
            train_map += average_precision_score(y_test, prediction) / len(train_contests)

        print("evaluating mAP for test set")

        test_true_label_list = []
        test_pred_list = []
        test_pred_contests = []
        for batch in test_dataloader:  ## If you have a DataLoader()  object to get the data.

            data = list(batch[0])
            targets = batch[1]  ## assuming that data loader returns a tuple of data and its targets
            top_ten_rank = batch[2].tolist()
            b_contests = batch[3].tolist()

            encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=50, add_special_tokens=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask).tolist()

            test_true_label_list += top_ten_rank
            test_pred_list += outputs
            test_pred_contests += b_contests
        contests_scores_df = pd.DataFrame(data={'contest': test_pred_contests, 'test_pred_list': test_pred_list,
                                                'test_true_label_list': test_true_label_list})

        test_map = 0
        for contest in test_contests:
            cur_contest_data = contests_scores_df[contests_scores_df['contest'] == contest]
            # y test is ordered
            y_test = cur_contest_data['test_true_label_list'].tolist()
            prediction = cur_contest_data['test_pred_list'].tolist()
            test_map += average_precision_score(y_test, prediction) / len(test_contests)

        print(f'epoch: {epoch}, train mAP: {train_map}, test mAP: {test_map})')

    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}

    # Save last, best and delete
    torch.save(ckpt, last)

    plt.figure()
    plt.plot([i for i in range(1, epochs + 1)], train_map_lst)
    plt.title('train mAP')
    plt.ylabel('mAP')
    plt.xlabel('epoch')
    plt.savefig('train_map.png')

    plt.figure()
    plt.plot([i for i in range(1, epochs + 1)], test_map_lst)
    plt.title('test mAP')
    plt.ylabel('mAP')
    plt.xlabel('epoch')
    plt.savefig('test_map.png')



if __name__ == '__main__':
    main()
