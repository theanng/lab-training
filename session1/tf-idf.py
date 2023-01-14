import os, re
import numpy as np
from collections import defaultdict


def gather_20newsgroup_data():
    path = 'data/20news-bydate/'
    #dirs = subfolders in 20news-bydate
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path + dir_name)]
    #train_dir, test_dir = (20news-bydate-train, 20news-bydate-test)
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])

    list_newgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newgroups.sort()

    with open('data/stop_words.txt') as f:
        stop_words = f.read().splitlines()
    
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path + filename)]
            files.sort()

            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    #remove stop words
                    words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                    #combine remaining words
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    train_data = collect_data_from(train_dir, list_newgroups)
    test_data = collect_data_from(test_dir, list_newgroups)
    
    full_data = train_data + test_data
    #export to text file
    with open('data/20news-bydate/20news-bydate-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('data/20news-bydate/20news-bydate-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('data/20news-bydate/20news-bydate-full-processed.txt','w') as f:
        f.write('\n'.join(full_data))

def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df>0
        return np.log10(corpus_size * 1/df)

    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
            
    words_idfs = [(word, compute_idf(document_freq, corpus_size)) for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq > 10 and not word.isdigit()]
    #sort in decreasing order
    words_idfs.sort(key = lambda t: t[-1], reverse= True)
    print("Vocabulary size:", len(words_idfs))
    with open('data/20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))

def get_tf_idf(data_path):
    with open('data/20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_IDs = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]), int(line.split('<fff>')[1]), line.split('<fff>')[2]) for line in f.read().splitlines()]
        
    data_tf_idfs = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])

        document_tf_idfs = list()
        sum_squares = 0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf = term_freq * idfs[word] / max_term_freq
            document_tf_idfs.append((word_IDs[word], tf_idf))
            sum_squares += tf_idf ** 2
        
        document_tf_idfs_normalized = [str(word_index) + ':' + str(tf_idf / np.sqrt(sum_squares)) for word_index, tf_idf in document_tf_idfs]

        sparse_rep = ' '.join(document_tf_idfs_normalized)
        data_tf_idfs.append((label, doc_id, sparse_rep))

    with open('data/20news-bydate/data-tf-idf.txt', 'w') as f:
        # export to a txt file, with each line being a document: news group, file name, tf-idf vector
        f.write('\n'.join(str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for label, doc_id, sparse_rep in data_tf_idfs))

gather_20newsgroup_data()
generate_vocabulary('data/20news-bydate/20news-bydate-full-processed.txt')
get_tf_idf('data/20news-bydate/20news-bydate-full-processed.txt')