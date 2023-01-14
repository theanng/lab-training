import numpy as np

def get_data(filename):
    with open(filename) as f:
        data = []
        for line in f:
            data.append([float(x) for x in line.split()])
        data = np.array(data)
        x = data[:, 1:-1]
        y = data[:, -1]
        y = np.array(y)
    return x, y

def normalize_and_add_ones(X):
    X = np.array(X)
    X_max = X.max(axis = 0)
    X_min = X.min(axis = 0)

    X_normalized = (X-X_min) / (X_max - X_min)

    ones = np.array([[1] for i in range (X_normalized.shape[0])])
    return np.column_stack((ones, X_normalized))

class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]

        W = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA * np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W

    def fit_gradient_descent(self, X_train, Y_train, LAMBDA, learning_rate, max_num_epoch = 100, batch_size = 128):
        W = np.random.randn(X_train.shape[1])
        last_loss = 10e8

        for ep in range(max_num_epoch):

            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))

            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index:index+batch_size]
                Y_train_sub = Y_train[index:index+batch_size]
                grad = X_train_sub.T.dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W = W - learning_rate*grad
            
            new_loss = self.compute_RSS(self.predict(W, X_train), Y_train)

            if np.abs(new_loss - last_loss) <= 1e-5:
                break
            last_loss = new_loss
        
        return W

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        Y_new = X_new.dot(W)
        return Y_new

    def compute_RSS(self, Y_new, Y_predicted):
        m = Y_new.shape[0]
        loss = 1/m * (np.sum((Y_new - Y_predicted)**2))
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            
            valid_ids = np.split(row_ids[: len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds :])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            
            aver_RSS = 0
            for i in range(num_folds):
                X_valid_fcv = X_train[valid_ids[i]]
                Y_valid_fcv = Y_train[valid_ids[i]]
                X_train_fcv = X_train[train_ids[i]]
                Y_train_fcv = Y_train[train_ids[i]]
                
                W = self.fit(X_train_fcv, Y_train_fcv, LAMBDA)
                aver_RSS += self.compute_RSS(Y_valid_fcv, self.predict(W, X_valid_fcv))
            
            return aver_RSS / num_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS
        
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=1000**2, LAMBDA_values=range(50))

        LAMBDA_values = [k * 1. / 1000 for k in range (max(0, (best_LAMBDA - 1) * 1000), (best_LAMBDA + 1) * 1000, 1)]
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values)
        return best_LAMBDA

if __name__ == '__main__':
    X, Y = get_data('data/deathrate.txt')

    X = normalize_and_add_ones(X)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print("Best LAMBDA:", best_LAMBDA)
    W_learned = ridge_regression.fit(X_train, Y_train, best_LAMBDA)
    Y_predicted = ridge_regression.predict(W_learned, X_test)

    print (ridge_regression.compute_RSS(Y_test, Y_predicted))
        