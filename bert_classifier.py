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


    def forward(self, ids, mask):
        output = self.bert(ids, mask)
        # print("AAA")
        # print(output[0].shape)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(output[0][:, 0, :].view(-1, 768))   # extract the 1st token's embeddings

        relu_output = self.relu(linear1_output)

        linear2_output = self.linear2(relu_output)

        return linear2_output.float()
