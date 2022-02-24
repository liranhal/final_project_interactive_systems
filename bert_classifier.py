from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch import nn


class BertHumorNetwork(nn.Module):
    def __init__(self):
        super(BertHumorNetwork, self).__init__()
        model_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(model_name, num_labels=2)

        self.linear1 = nn.Linear(768, 256)
        self.relu = nn.LeakyReLU(0.3)
        self.linear2 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        output = self.bert(ids, mask)
        # print("AAA")
        # print(output[0].shape)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(output[0][:, 0, :].view(-1, 768))   # extract the 1st token's embeddings

        relu_output = self.relu(linear1_output)

        linear2_output = self.linear2(relu_output)

        sigmoid_output = 3 * self.sigmoid(linear2_output)

        sigmoid_output = sigmoid_output.squeeze()

        return sigmoid_output.float()


class MixedBertHumorNetwork(nn.Module):
    def __init__(self):
        super(MixedBertHumorNetwork, self).__init__()
        model_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(model_name, num_labels=2)

        self.bertlinear1 = nn.Linear(768, 256)

        self.dropout = nn.Dropout()

        self.relu = nn.LeakyReLU(0.3)

        self.baseline_linear1 = nn.Linear(12, 12)

        self.linear2 = nn.Linear(256 + 12, 1)

        self.sigmoid = nn.Sigmoid()

        for param in self.bert.parameters():
            param.requires_grad = False

        self.bertlinear1.requires_grad = True

        self.baseline_linear1.requires_grad = True

        self.linear2.requires_grad = True

    def forward(self, ids, mask, baseline_vec):
        baseline_vec = baseline_vec / torch.norm(baseline_vec)
        output = self.bert(ids, mask)
        # print("AAA")
        # print(output[0].shape)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        bert_linear1_output = self.bertlinear1(output[0][:, 0, :].view(-1, 768))   # extract the 1st token's embeddings

        bert_linear1_output = self.dropout(bert_linear1_output)

        relu_bert_output = self.relu(bert_linear1_output)

        baseline_linear_output = self.baseline_linear1(baseline_vec)

        baseline_linear_output = self.dropout(baseline_linear_output)

        relu_baseline_output = self.relu(baseline_linear_output)

        cat_vec = torch.cat((relu_baseline_output, relu_bert_output), 1)

        linear2_output = self.linear2(cat_vec)

        sigmoid_output = 3 * self.sigmoid(linear2_output)

        sigmoid_output = sigmoid_output.squeeze()

        return sigmoid_output.float()

