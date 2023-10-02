import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        X = X - np.mean(X,axis=0,keepdims=True)
        variance_matrix = X.T@X
        self.eig,self.vec=np.linalg.eigh(variance_matrix)
        
        return None
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        self.X = X @ self.vec[:,:-(self.n_components+1):-1]
        
        return self.X

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        self.w = np.zeros(X.shape[1])
        self.b = 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            j = np.random.randint(X.shape[0])
            x_j,y_j = X[j],y[j]
            
            margin = y_j * (np.dot(x_j,self.w)+self.b)
            
            if margin < 1:
                grad_w = self.w - (C* y_j *x_j)
                grad_b = -C *y_j
                
            else :
                grad_w = self.w
                grad_b = 0
               
            self.w -= learning_rate*(grad_w)
            self.b -= learning_rate*(grad_b) 
            
     
            
            
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        y_predict = np.dot(X,self.w) + self.b
        
        return y_predict

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, learning_rate,num_iters ,C) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        
        
        for i in range(self.num_classes):
            
            y_i = np.where(y == i, 1,-1)
            self.models[i].fit(X,y_i,learning_rate,num_iters,C)
            
    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0],self.num_classes))
        for i in range(self.num_classes):
            svm_i = self.models[i]
            scores[:,i] = svm_i.predict(X) 
            
        return np.argmax(scores,axis=1)   
        

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        precision_score = np.zeros(self.num_classes)
        
        y_pred = self.predict(X)
        for i in range(self.num_classes):
            true_positives = np.sum((y_pred == i) & (y == i))
            total_positives = np.sum(y_pred == i)
            precision_score[i] = (true_positives / total_positives) if total_positives != 0 else 0
        
        return np.mean(precision_score)
    
    def recall_score(self, X, y) -> float:
        true_positives = np.zeros(self.num_classes)
        false_negatives= np.zeros(self.num_classes)
        
        recall_per_class = np.zeros(self.num_classes)
        y_pred = self.predict(X)
        for i in range(self.num_classes):
            
            true_positives[i] = np.sum((y_pred == i) & (y == i))
            false_negatives[i] = np.sum((y_pred != i) & (y == i))
            recall_per_class[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
            
        return np.mean(recall_per_class)
            
    
 
    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X,y)
        recall = self.recall_score(X,y)
        F1 = 2 * (precision * recall) / (precision + recall)
        
        return F1
