import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC


def load_data():
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        
        indices_and_tfidfs = sparse_r_d.split()
        for index_and_tfidf in indices_and_tfidfs:
            index = int(index_and_tfidf.split(':')[0])
            tfidf = float(index_and_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)    
            
    with open('data-tf-idf.txt') as f:
        d_lines = f.read().splitlines()
    with open("words_idfs.txt") as f:
        vocab_size = len(f.read().splitlines())

    labels = []
    data = []
    label_count = defaultdict(int)
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        label_count[label] += 1
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    
    return data, np.array(labels)

def clustering_with_KMeans():
    data, labels = X_train, y_train
    #use csr_matrix to creat a sparse matrix with efficient row slicing
    X = csr_matrix(data)
    print('=======')
    kmeans = KMeans(n_clusters=20, init='random', 
    n_init=5,   #number of time that kmeans runs with differently initialized centroids
    tol=1e-3,   #threshold for acceptable minimum error decrease
    random_state=2018   #set to get deterministic results
    ).fit(X)
    labels = kmeans.labels_

    predicted_y = kmeans.predict(X_test)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=y_test)
    print("Accuracy: ",accuracy)

def classifying_with_linear_SVMs():
    classifier = LinearSVC(
        C=10.0,  #penalty coefficient
        tol=0.001,  #tolerance for stopping criteria
        verbose=True,   #whether prints out logs or not
    )
    classifier.fit(X_train, y_train)

    predicted_y = classifier.predict(X_test)
    accuracy = compute_accuracy(predicted_y, y_test)
    print("Accuracy:", accuracy)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / expected_y.size
    return accuracy

X,y = load_data()
#training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 30)
X_train=csr_matrix(X_train)


classifying_with_linear_SVMs()