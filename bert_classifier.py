from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch import nn


class BertHumorNetwork(nn.Module):
    def __init__(self):
        super(BertHumorNetwork, self).__init__()
        model_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(model_name, num_labels=2)

        self.linear1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 1)


    def forward(self, ids, mask):
        sequence_output, pooled_output = self.bert(ids, attention_mask=mask)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(sequence_output[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings

        linear2_output = self.linear2(linear1_output)

        return linear2_output
