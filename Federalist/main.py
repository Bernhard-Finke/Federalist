from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn

"""
Code to predict the authorship of the disputed Federalist Papers
Uses a form of term frequency inverse-document frequency
"""

url = "https://www.gutenberg.org/files/1404/1404-h/1404-h.htm#link2H_4_0001"
site = requests.get(url)

soup = BeautifulSoup(site.text, features="lxml")
p_tags = soup.find_all("p")


paper_list = []
paper_index = -1

for text in p_tags:
    # breaks up each text at the phrase which marks the beginning of each paper
    if "To the People of the State of New York" in text.get_text():
        if paper_index != -1:
            del paper_list[paper_index][-1]
        paper_index += 1
        paper_list.append([])
        continue

    if paper_index == -1:
        continue

    cleaned_text = []
    string_remove = '!"#()*, -./:;<=>?\_`{|}~0123456789'
    remove_punctuation = str.maketrans('', '', string_remove)
    [cleaned_text.append(word.lower().translate(remove_punctuation)) for word in text.get_text().split()]

    [paper_list[paper_index].append(paragraph) for paragraph in cleaned_text]


vocab_by_paper = []
empty_vocab = {}
total_vocab = {}


for i in range(85):
    for word in paper_list[i]:
        if word not in empty_vocab:
            empty_vocab.update({word: 0})
            total_vocab.update({word: 1})
        else:
            total_vocab.update({word: total_vocab[word]})


for j in range(85):
    vocab_by_paper.append(empty_vocab.copy())
    [vocab_by_paper[j].update({word: vocab_by_paper[j][word]+1}) for word in paper_list[j]]


disputed_vocab = {}
training_data = []
training_labels = []

author_list = ['h', 'j', 'j', 'j', 'j', 'h', 'h', 'h', 'h', 'm', 'h', 'h', 'h', 'm', 'h', 'h', 'h', 'm', 'm', 'm',
               'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'm', 'm', 'm', 'm',
               'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'h', 'h',
               'h', 'd', 'd', 'j', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h',
               'h', 'h', 'h', 'h', 'h']

tf_idf = lambda ind, w, tot: (vocab_by_paper[ind][w]/tot)*np.log(total_vocab[w]/68)

for p in range(85):
    sum_words = sum(vocab_by_paper[p].values())
    if author_list[p] == 'd':
        disputed_vocab.update({str(p+1): []})
        disputed_vocab[str(p+1)].extend([tf_idf(p, word, sum_words) for word in vocab_by_paper[p].keys()])
        continue
    training_data.append([])
    training_data[len(training_data)-1].extend([tf_idf(p, word, sum_words) for word in vocab_by_paper[p].keys()])
    training_labels.extend(author_list[p])

correct = 0
neigh = knn(n_neighbors=3)
neigh.fit(training_data, training_labels)

predictions = []

for paper_key in disputed_vocab.keys():
    if neigh.predict(np.reshape(disputed_vocab[paper_key], (1, -1))) == ['m']:
        correct += 1
    predictions.append(neigh.predict(np.reshape(disputed_vocab[paper_key], (1, -1))))


print("The K-nearest neighbors algorithm correctly predicts " + str(correct) + " of the 12 disputed Federalist Papers.")