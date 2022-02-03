import nltk
from sklearn.feature_extraction.text import CountVectorizer
import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, automated_readability_index
from nltk.lm.preprocessing import pad_both_ends, flatten, pad_sequence
from nltk.lm import Vocabulary
from nltk.corpus import brown
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import pandas as pd
import numpy as np
import re
from nltk.util import ngrams


GLOVE_PATH = '/home/student/pretrained_embds/glove.6B.50d.txt'


def perplexity_score_GPT(sentence, tokenizer, model):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def cosine_sim(v, w):
    return (v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))


def load_comp_csv(comp_path):
    dir_files = os.listdir(comp_path)
    comp_files = []

    for f_name in dir_files:
        if '_summary_KLUCB.csv' in f_name:
            df = pd.read_csv(f'{comp_path}/{f_name}')
            comp_files.append(df)

    merged_df = pd.concat(comp_files, axis=0)
    return merged_df.reset_index()

def preprocess(x):
    x = str(x).lower()
    return re.sub(r'[^\w\s]', '', x)

def preprocessed_list(lst):
    return [preprocess(x) for x in lst]

def n_gram(caption, n):
    padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>",
                                    pad_right=True, right_pad_symbol="</s>", n=3))
    list(ngrams(padded_sent, n=3))


def create_dataset(comp_path, context_anomaly_path, target_path):
    merged_df = load_comp_csv(comp_path)
    print(merged_df.columns)
    print(merged_df.head())
    merged_df['processed_captions'] = merged_df['caption'].apply(preprocess)
    merged_df['mask'] = merged_df['processed_captions'].apply(lambda x: 5 <= len(x.split()) <= 10)
    merged_df = merged_df[merged_df['mask']]
    print(merged_df.head())


    # computing the perplexity of the sentence

    # model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    # model.eval()
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    #
    # print([perplexity_score(i, tokenizer, model) for i in data])


    #TODO: change to brown
    brown.sents(categories=['news'])

    brown_data = list(map(preprocessed_list, brown.sents(categories=['news'])))


    brown_data_lst = [' '.join(b) for b in brown_data]
    train_data4, vocab4 = padded_everygram_pipeline(4, brown_data)
    train_data3, vocab3 = padded_everygram_pipeline(3, brown_data)
    train_data2, vocab2 = padded_everygram_pipeline(2, brown_data)
    train_data1, vocab1 = padded_everygram_pipeline(1, brown_data)

    # tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in brown_data_lst]
    # train_data = [nltk.bigrams(pad_both_ends(text[0], n=2)) for text
    #               in tokenized_text]
    #
    # vocab = list(flatten(pad_both_ends(sent, n=2) for sent in tokenized_text))
    #
    # padded_vocab = Vocabulary(vocab)

    lm4 = MLE(4)
    lm3 = MLE(3)
    lm2 = MLE(2)
    lm1 = MLE(1)

    lm4.fit(train_data4, vocab4)
    lm3.fit(train_data3, vocab3)
    lm2.fit(train_data2, vocab2)
    lm1.fit(train_data1, vocab1)

    test = [('fulton',),  ('grand', )]

    merged_df['unigrams'] = merged_df['processed_captions'].apply(lambda x: list(ngrams(x.split(), n=1)))
    merged_df['bigrams'] = \
        merged_df['processed_captions'].apply(lambda x: list(ngrams(pad_sequence(x.split(), n=2, left_pad_symbol='<s>', pad_left=True), n=2)))
    merged_df['trigrams'] = \
        merged_df['processed_captions'].apply(lambda x: list(ngrams(pad_sequence(x.split(), n=3, left_pad_symbol='<s>', pad_left=True), n=3)))
    merged_df['fourgrams'] = \
        merged_df['processed_captions'].apply(lambda x: list(ngrams(pad_sequence(x.split(), n=4, left_pad_symbol='<s>', pad_left=True), n=4)))
    print(merged_df.head())

    print(merged_df['unigrams'].apply(lambda  x: len(x)).min())

    merged_df['unigrams_perplexity'] = merged_df['unigrams'].apply(lambda x: lm1.perplexity(x))
    merged_df['bigrams_perplexity'] = merged_df['bigrams'].apply(lambda x: lm2.perplexity(x))
    merged_df['trigrams_perplexity'] = merged_df['trigrams'].apply(lambda x: lm3.perplexity(x))
    merged_df['fourgrams_perplexity'] = merged_df['fourgrams'].apply(lambda x: lm4.perplexity(x))

    print(merged_df.head())

    # print(test)
    print(lm4.perplexity(test))
    print(lm3.perplexity(test))
    print(lm2.perplexity(test))
    print(lm1.perplexity(test))


    # readability scores

    merged_df['flesch_readablity'] = merged_df['processed_captions'].apply(lambda x: flesch_reading_ease(x))
    merged_df['automated_readability'] = merged_df['processed_captions'].apply(lambda x: automated_readability_index(x))

    print(merged_df.head())


    # sentiment scores

    sia = SentimentIntensityAnalyzer()

    merged_df['positive_sent'] = merged_df['processed_captions'].apply(lambda x: sia.polarity_scores(x)['pos'])
    merged_df['neutral_sent'] = merged_df['processed_captions'].apply(lambda x: sia.polarity_scores(x)['neu'])
    merged_df['negative_sent'] = merged_df['processed_captions'].apply(lambda x: sia.polarity_scores(x)['neg'])
    print(merged_df.head())

    # third_person_pronouns = ['he', him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves]

    definite_article = 'the'

    merged_df['count_definite_article'] = merged_df['processed_captions']\
        .apply(lambda sent: sent.split(' ').count(definite_article))

    indefinite_articles = ['a', 'an']

    eps = 1e-5
    merged_df['count_indefinite_article'] = merged_df['processed_captions']\
        .apply(lambda sent: sum([sent.split(' ').count(article) + eps for article in indefinite_articles]))

    merged_df['definite_vs_indefinite'] = merged_df['count_definite_article'] / merged_df['count_indefinite_article']
    print(merged_df.head())




    # BOW features

    bow_vectorizer = CountVectorizer()
    bow_vectors = bow_vectorizer.fit_transform(merged_df['processed_captions'].tolist())

    print(bow_vectors)
    # Word2Vec features

    glove_dict = load_glove_model(GLOVE_PATH)

    anomaly_context_df = pd.read_csv(context_anomaly_path)

    merged_df = merged_df.merge(anomaly_context_df, how='inner', on='contest')

    merged_df['context_words_list'] = merged_df[['context_word_1', 'context_word_2', 'context_word_3']].tolist()

    merged_df['anomaly_words_list'] = merged_df[['anomaly_word_1', 'anomaly_word_2', 'anomaly_word_3']].tolist()

    joke_words_list = merged_df['processed_captions'].apply(lambda x: x.split()).tolist()
    anomaly_words_list = merged_df['anomaly_words_list'].tolist()
    context_words_list = merged_df['context_words_list'].tolist()

    sim_features_lst = []

    for joke_set, context_set, anomaly_set in zip(joke_words_list, context_words_list, anomaly_words_list):
        context_sim_lst = [max(cosine_sim(glove_dict[j], glove_dict[c]) for c in context_set if c) for j in joke_set]
        anomaly_sim_lst = [max(cosine_sim(glove_dict[j], glove_dict[a]) for a in anomaly_set if a) for j in joke_set]

        context_sim = max(context_sim_lst)
        anomaly_sim = max(anomaly_sim_lst)

        max_absolute_diff = max(abs(c - a) for c, a in zip(context_sim_lst, anomaly_sim_lst))
        average_absolute_diff = sum(abs(c - a) for c, a in zip(context_sim_lst, anomaly_sim_lst))/len(joke_set)

        sim_features_lst.append((context_sim, anomaly_sim, max_absolute_diff, average_absolute_diff))

    merged_df.to_csv(target_path, index=False, header=False)


if __name__ == '__main__':
    create_dataset('/home/student/Desktop/caption-contest-data/contests/summaries', 'context_anomaly_words.csv',
                   'baseline_features.csv')
