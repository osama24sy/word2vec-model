import numpy as np
import load_data
import word2vec

settings = {}
settings['n'] = 2                  # dimension of word embeddings
settings['window_size'] = 5         # context window before target word
# settings['min_count'] = 2         # minimum word count
settings['epochs'] = 3            # number of training epochs
# settings['neg_samp'] = 10         # number of negative words to use during training
settings['learning_rate'] = 0.05    # learning rate
settings['num_word'] = 50         # Number of the most frequent unique words
np.random.seed(0)                   # set the seed for reproducibility

path = './example.txt'

corpus_raw, data, word, info = load_data.load_data(path)

# INITIALIZE W2V MODEL
w2v = word2vec.Word2vec(settings)

# generate training data
training_data = w2v.generate_training_data(data)

# train word2vec model
w2v.train(training_data)