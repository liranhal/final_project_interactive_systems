import nltk
from sklearn.feature_extraction.text import CountVectorizer
import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.lm import Vocabulary
from nltk.corpus import brown
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np


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


def create_data():
    data = ['there is a book on the desk',
            'there is a plane on the desk',
            'i read a cute book',
            'i read an interesting book']

    # computing the perplexity of the sentence

    # model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    # model.eval()
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    #
    # print([perplexity_score(i, tokenizer, model) for i in data])


    #TODO: change to brown
    brown.sents(categories=['news'])

    brown_data = brown.sents(categories=['news'])

    print(brown_data)

    brown_data_lst = [' '.join(b) for b in brown_data]
    train_data, vocab = padded_everygram_pipeline(2, brown_data)

    # tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in brown_data_lst]
    # train_data = [nltk.bigrams(pad_both_ends(text[0], n=2)) for text
    #               in tokenized_text]
    #
    # vocab = list(flatten(pad_both_ends(sent, n=2) for sent in tokenized_text))
    #
    # padded_vocab = Vocabulary(vocab)

    lm = MLE(2)

    lm.fit(train_data, vocab)
    test = [('Fulton', 'County'), ('Grand', )]

    print(test)
    print(lm.perplexity(test))

    exit()

    # readability scores

    textstat.flesch_reading_ease(data)
    textstat.automated_readability_index(data)

    # sentiment scores

    sia = SentimentIntensityAnalyzer()
    print([sia.polarity_scores(sen) for sen in data])

    # third_person_pronouns = ['he', him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves]

    definite_article = 'the'

    count_definite_article = np.array([sent.split(' ').count(definite_article) for sent in data])

    indefinite_articles = ['a', 'an']

    eps = 1e-5
    count_indefinite_article = np.array([sum([sent.split(' ').count(article) + eps for article in indefinite_articles])
                                for sent in data])

    definite_vs_indefinite = count_definite_article / count_indefinite_article




    # BOW features

    bow_vectorizer = CountVectorizer()
    bow_vectors = bow_vectorizer.fit_transform(data)

    # Word2Vec features

    glove_dict = load_glove_model(GLOVE_PATH)

    joke_words_list = []
    anomaly_words_list = []
    context_words_list = []

    sim_features_lst = []

    for joke_set, context_set, anomaly_set in zip(joke_words_list, context_words_list, anomaly_words_list):
        context_sim_lst = [max(cosine_sim(glove_dict[j], glove_dict[c]) for c in context_set) for j in joke_set]
        anomaly_sim_lst = [max(cosine_sim(glove_dict[j], glove_dict[a]) for a in context_set) for j in joke_set]

        context_sim = max(context_sim_lst)
        anomaly_sim = max(anomaly_sim_lst)

        max_absolute_diff = max(abs(c - a) for c, a in zip(context_sim_lst, anomaly_sim_lst))
        average_absolute_diff = sum(abs(c - a) for c, a in zip(context_sim_lst, anomaly_sim_lst))/len(joke_set)

        sim_features_lst.append((context_sim, anomaly_sim, max_absolute_diff, average_absolute_diff))



if __name__ == '__main__':
    create_data()