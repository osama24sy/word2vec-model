import nltk
import time

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def tokenizer(text) :
    start = time.time() # Start timer
    sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences

    data = []
    words = []
    switch = 0
    switch_sentence = []
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence.lower())
        sent = [word for word in tokenized_text if (word.isalpha() or word == "n't")]

        # If the switch flipped add the sentence to the temp list
        if switch == 1:
            for word in sent:
                switch_sentence.append(word)
                words.append(word)
            if len(switch_sentence) >= 5: # If the temp list is bigger than five we flip the switch
                switch = 0
                data.append(switch_sentence)
                switch_sentence = []
            continue

        # If the sentences is smaller than 5 flip the switch and add it to a temp list
        if len(sent) <= 5:
            switch = 1
            for word in sent:
                switch_sentence.append(word)
                words.append(word)
            continue
        # If the sent is bigger than 5 and the switch is off
        data.append(sent)
        for word in sent:
            words.append(word)

    end = time.time() # End timer

    info = {
        "wordCount": len(words),
        "sentenceCount": len(data),
        "dictSize": len(set(words)),
        "runtime": end - start
    }
    return data, words, info

def load_data(path) :
    # Import file
    corpus_raw = ""
    text_data = open(path, "r", encoding='unicode_escape')
    for line in text_data.readlines():
        corpus_raw += line
    text_data.close()
    
    print(f"The data has been successfully loaded from {path}")
    
    corpus_raw = corpus_raw.lower()

    data, words, info = tokenizer(corpus_raw)

    print(f"{path} has been successfully processed and tokenized\n",
          f"Total number of words: {info['wordCount']}\n",
          f"Total sentence count: {info['sentenceCount']}\n",
          f"The dictionary size is : {info['dictSize']}\n", 
          f"The runtime is {info['runtime']} seconds\n")

    return corpus_raw, data, words, info
