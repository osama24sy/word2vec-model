import numpy as np
import re
from collections import Counter
import csv
import pandas as pd


class Word2vec:
    def __init__(self, settings):
        self.n = settings["n"]
        self.eta = settings["learning_rate"]
        self.epochs = settings["epochs"]
        self.window = settings["window_size"]
        self.num_word = settings["num_word"]
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, corpus):

        # GENERATE WORD COUNTS
        word_counts = dict(
            sorted(
                Counter(sum(corpus, [])).items(), key=lambda item: item[1], reverse=True
            )
        )

        self.v_count = min(len(word_counts.keys()), self.num_word)

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = list(word_counts.keys())
        self.word_index = dict(
            (word, i) for i, word in enumerate(self.words_list[: self.v_count])
        )
        self.index_word = dict(
            (i, word) for i, word in enumerate(self.words_list[: self.v_count])
        )

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                # w_target  = sentence[i]
                if sentence[i] in self.word_index:
                    w_target = self.word_index[sentence[i]]

                    # CYCLE THROUGH CONTEXT WINDOW
                    w_context = []
                    for j in range(i - self.window, i):
                        if (
                            j != i
                            and j <= sent_len - 1
                            and j >= 0
                            and (sentence[j] in self.word_index)
                        ):
                            w_context.append(self.word_index[sentence[j]])
                    if w_context != []:
                        training_data.append([w_target, w_context])
        return np.array(training_data, dtype=object)

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_vec[word] = 1
        return word_vec

    def attention(self, x):
        weights = []
        for vec in x:
            dots = np.dot(self.w1[x], self.w1[vec].T)
            weights.append(np.dot(self.w1[x].T, self.softmax(dots)))

        return weights

    # FORWARD PASS
    def forward_pass(self, x, att=False):
        # h = np.dot(self.w1.T, x)
        if att:
            weight1 = self.attention(x)
        else:
            weight1 = self.w1[x]
        h = np.sum(weight1, axis=0)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw1 = np.dot(self.w2, e)
        dl_dw2 = np.outer(h, e)

        # UPDATE WEIGHTS
        for i in x:
            self.w1[i] = self.w1[i] - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        # print(self.w1.shape, dl_dw1.shape, self.w2.shape, dl_dw2.shape)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(
            -0.8, 0.8, (self.v_count, self.n)
        )  # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # context matrix
        # train_leng= len(training_data)
        # CYCLE THROUGH EACH EPOCH
        self.losses = []
        for i in range(0, self.epochs):
            # counter =0
            self.loss = 0
            # CYCLE THROUGH EACH TRAINING SAMPLE
            for t, c in training_data:

                w_t = self.word2onehot(t)
                # counter+=1
                # print(counter/train_leng)
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(c)

                # CALCULATE ERROR
                EI = np.subtract(y_pred, w_t)

                # BACKPROPAGATION
                self.backprop(EI, h, c)

                # CALCULATE LOSS
                self.loss += -u[t] + np.log(np.sum(np.exp(u)))
                # self.loss += -np.sum([u[j] for j in c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            # self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
            self.losses.append(self.loss)
            print("EPOCH:", i, "LOSS:", self.loss)
        pass

    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def predict(self, sentence, top_n, att=False):
        words_vec = [self.word_index[word] for word in sentence]
        y_pred, h, u = self.forward_pass(words_vec, att)
        pred_words = dict(enumerate(y_pred))
        sorted_pred_words = sorted(
            pred_words.items(), key=lambda item: item[1], reverse=True
        )

        for i, pred in sorted_pred_words[:top_n]:
            print(self.index_word[i], pred)
        pass

    # Save the word embeddings in a csv file
    def save_model(self, name=None):

        out = [[] for i in range(self.num_word)]

        for word, index in self.word_index.items():
            out[index].append(word)
            for embed in self.w1[index]:
                out[index].append(embed)

        if name != None:
            csv_file_name = name + "_embeddings.csv"
        else:
            csv_file_name = (
                "dim"
                + str(self.n)
                + "_lr"
                + str(self.eta)
                + "_ep"
                + str(self.epochs)
                + "_w"
                + str(self.window)
                + "_n"
                + str(self.num_word)
                + "_embeddings.csv"
            )

        # Write the NumPy array to the CSV file
        with open(csv_file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write each row of the NumPy array to the CSV file
            for row in out:
                csv_writer.writerow(row)

        print(
            f"The model embeddings matrix has been successfully saved into {csv_file_name}."
        )
        pass

    # Import word embeddings
    def import_model(self, path):

        data = pd.read_csv(path, header=None)

        self.word_index = {}
        self.w1 = np.empty((data.shape[0], data.shape[1] - 1))
        for index, row in data.iterrows():
            self.word_index[row[0]] = index
            self.w1[index] = row[1:]

        print(
            f"The model embeddings matrix has been successfully imported from {path}."
        )
        pass

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

        pass
