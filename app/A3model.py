import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class LogisticRegression(object):
    
    def __init__(self, regularization, k, n, method, alpha = 0.001, max_iter=5000):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m  
        loss += self.regularization(self.W, m) # add regularization term to the loss
        error = h - Y
        grad = self.softmax_grad(X, error) 
        grad += self.regularization.derivative(self.W, m) # add regularization term to the gradient
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, cls):
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def recall(self, y_true, y_pred, cls):
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def f1_score(self, y_true, y_pred, cls):
        prec = self.precision(y_true, y_pred, cls)
        rec = self.recall(y_true, y_pred, cls)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    def macro_precision(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.precision(y_true, y_pred, cls) for cls in classes])

    def macro_recall(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.recall(y_true, y_pred, cls) for cls in classes])

    def macro_f1_score(self, y_true, y_pred):
        classes = np.unique(y_true)
        return np.mean([self.f1_score(y_true, y_pred, cls) for cls in classes])

    def weighted_precision(self, y_true, y_pred):
        classes, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        return np.sum([self.precision(y_true, y_pred, cls) * (count / total) for cls, count in zip(classes, counts)])

    def weighted_recall(self, y_true, y_pred):
        classes, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        return np.sum([self.recall(y_true, y_pred, cls) * (count / total) for cls, count in zip(classes, counts)])

    def weighted_f1_score(self, y_true, y_pred):
        classes, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        return np.sum([self.f1_score(y_true, y_pred, cls) * (count / total) for cls, count in zip(classes, counts)])

    def classification_report(self, y_true, y_pred, digits=2):
        classes, counts = np.unique(y_true, return_counts=True)
        total_support = np.sum(counts)

        report = {}

        # Per-class metrics
        for cls, count in zip(classes, counts):
            cls_str = str(cls)
            p = self.precision(y_true, y_pred, cls)
            r = self.recall(y_true, y_pred, cls)
            f = self.f1_score(y_true, y_pred, cls)
            report[cls_str] = {
                "precision": round(p, digits),
                "recall": round(r, digits),
                "f1-score": round(f, digits),
                "support": int(count)
            }

        # Accuracy
        acc = self.accuracy(y_true, y_pred)
        report["accuracy"] = {
            "precision": "",
            "recall": "",
            "f1-score": round(acc, digits),
            "support": int(total_support)
        }

        # Macro avg
        macro_p = self.macro_precision(y_true, y_pred)
        macro_r = self.macro_recall(y_true, y_pred)
        macro_f = self.macro_f1_score(y_true, y_pred)
        report["macro avg"] = {
            "precision": round(macro_p, digits),
            "recall": round(macro_r, digits),
            "f1-score": round(macro_f, digits),
            "support": int(total_support)
        }

        # Weighted avg
        weighted_p = self.weighted_precision(y_true, y_pred)
        weighted_r = self.weighted_recall(y_true, y_pred)
        weighted_f = self.weighted_f1_score(y_true, y_pred)
        report["weighted avg"] = {
            "precision": round(weighted_p, digits),
            "recall": round(weighted_r, digits),
            "f1-score": round(weighted_f, digits),
            "support": int(total_support)
        }

        return pd.DataFrame(report).T

    
# Define Ridge regularization and No regularization classes
class RidgePenalty:
    def __init__(self, l2):
        self.l2 = l2

    def __call__(self, theta, m):
        return (self.l2 / (2 * m)) * np.sum(np.square(theta))

    def derivative(self, theta, m):
        return (2 * self.l2 / m) * theta

class NoPenalty:
    def __call__(self, theta, m=None):
        return 0.0

    def derivative(self, theta, m=None):
        return np.zeros_like(theta)

class Ridge(LogisticRegression):
    def __init__(self, k, n, method, alpha, l2):
        self.regularization = RidgePenalty(l2)
        super().__init__(self.regularization, k, n, method, alpha)

class SimpleLogistic(LogisticRegression):
    def __init__(self, k, n, method, alpha):
        self.regularization = NoPenalty()
        super().__init__(self.regularization, k, n, method, alpha)