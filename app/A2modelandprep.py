import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle


# Custom classes from the notebook to support A2 model loading
class NoRegularization:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return np.zeros_like(theta)

# Lasso Regularization (L1)
class LassoPenalty:
    def __init__(self, l):
        self.l = l # lambda value
    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))
    def derivation(self, theta):
        return self.l * np.sign(theta)

# Ridge Regularization (L2)
class RidgePenalty:
    def __init__(self, l):
        self.l = l # lambda value
    def __call__(self, theta):
        return self.l * np.sum(theta**2)
    def derivation(self, theta):
        return 2 * self.l * theta

# Custom LinearRegression class from notebook
class LinearRegression(object):
    # In this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)

    def __init__(self, 
                 regularization = None, 
                 lr=0.001, 
                 method='batch', 
                 num_epochs=500, 
                 batch_size=50, 
                 cv=kfold, 
                 init_method='zeros', 
                 use_momentum=False, 
                 momentum=0.9,
                 poly_degree=1):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization if regularization is not None else NoRegularization()

        # Addition of parameters
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.poly_degree = poly_degree

        # Check if momentum is used, and momentum value is between 0 and 1
        if self.use_momentum and not (0 < self.momentum < 1):
            raise ValueError("Momentum value must be between 0 and 1.")

    def mse(self, ytrue, ypred):
        # Verify if it is scalar or array
        if np.isscalar(ytrue):
            return (ypred - ytrue) ** 2
        else:
            return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # Add a function to compute R-squared score
    def r2(self, ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum() # Residual sum of squares
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum() # Total sum of squares
        if ss_tot == 0:
            return 0  # Avoid division by zero; return 0 if variance is zero
        return 1 - (ss_res/ss_tot)

    def poly_features(self, X, degree):
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X_train, y_train):
        # Create a list of kfold scores
        self.kfold_scores = list()

        # kfold.split in the sklearn.....
        # 3 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            # Reset val loss (Move it inside the fold loop due to early stopping)
            self.val_loss_old = np.inf
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            # Add polynomial features if degree > 1
            if self.poly_degree > 1:
                X_cross_train = self.poly_features(X_cross_train, self.poly_degree)
                X_cross_val = self.poly_features(X_cross_val, self.poly_degree)

            # Find the number of features in the dataset
            n_features = X_cross_train.shape[1]
            
            # Add xavier weights initialization
            if self.init_method == 'xavier':
                limit = np.sqrt(1.0/n_features)
                lower,upper = -limit, limit
                number = np.random.rand(n_features)
                self.theta = lower + number * (upper - lower)

            else: # Set default initialization to zeros
                self.theta = np.zeros(n_features)  

            # init once per fold
            self.prev_step = np.zeros_like(self.theta)

            # Define X_cross_train as only a subset of the data
            # How big is this subset?  => mini-batch size ==> 50

            # One epoch will exhaust the WHOLE training set
            # Note: Removed mlflow logging for app compatibility
            for epoch in range(self.num_epochs):
            
                # With replacement or no replacement
                # with replacement means just randomize
                # with no replacement means 0:50, 51:100, 101:150, ......300:323
                # Shuffle your index
                perm = np.random.permutation(X_cross_train.shape[0])
                        
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]
                
                if self.method == 'sto':
                    for batch_idx in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[batch_idx].reshape(1, -1) # (11,) ==> (1, 11) ==> (m, n)
                        y_method_train = y_cross_train[batch_idx] 
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                        # batch_idx = 0, 50, 100, 150
                        X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                        train_loss = self._train(X_method_train, y_method_train)
                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)
                val_r2_new = self.r2(y_cross_val, yhat_val)

                # Early stopping
                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new
        
            self.kfold_scores.append(val_loss_new)
            print(f"Fold {fold}: {val_loss_new}")
        
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)

        # Update with momentum if enabled
        if self.use_momentum:
            step = self.lr * grad
            update = self.momentum * self.prev_step - step
            self.theta += update
            self.prev_step = step # Store the current step for the next iteration
        else: # Standard gradient descent update
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)

    def predict(self, X, is_polynomial=False):
        if is_polynomial:
            X = self.poly_features(X, self.poly_degree)
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  # Remind that theta is (w0, w1, w2, w3, w4.....wn)
                               # w0 is the bias or the intercept
                               # w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]

# Define Lasso classes inheriting from LinearRegression
class Lasso(LinearRegression):
    def __init__(self, l=0.01, **kwargs):
        super().__init__(regularization=LassoPenalty(l), **kwargs)

# Define Ridge classes inheriting from LinearRegression
class Ridge(LinearRegression):
    def __init__(self, l=0.01, **kwargs):
        super().__init__(regularization=RidgePenalty(l), **kwargs)

class Polynomial(LinearRegression):
    def __init__(self, degree=2, **kwargs):
        super().__init__(poly_degree=degree, **kwargs)

# ManualPreprocessor class from the notebook
class ManualPreprocessor:
    def __init__(self, num_med_cols, num_mean_cols, cat_cols, drop_first=True):
        self.num_med_cols = list(num_med_cols)
        self.num_mean_cols = list(num_mean_cols)
        self.cat_cols = list(cat_cols)
        self.drop_first = drop_first
        # learned params
        self.medians_ = {}
        self.means_ = {}
        self.num_mean_for_scale_ = {}
        self.num_std_for_scale_ = {}
        self.cat_categories_ = {}
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame):
        X = X.copy()

        # 1) impute stats
        for c in self.num_med_cols:
            if c in X.columns:
                self.medians_[c] = X[c].median()
        for c in self.num_mean_cols:
            if c in X.columns:
                self.means_[c] = X[c].mean()

        # 2) impute to compute scaler on train
        for c in self.num_med_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self.medians_[c])
        for c in self.num_mean_cols:
            if c in X.columns:
                X[c] = X[c].fillna(self.means_[c])

        # 3) scaler stats (column-wise)
        num_all = self.num_med_cols + self.num_mean_cols
        for c in num_all:
            if c in X.columns:
                self.num_mean_for_scale_[c] = X[c].mean()
                self.num_std_for_scale_[c] = X[c].std(ddof=0)
                # Ensure std is not zero
                if self.num_std_for_scale_[c] == 0:
                    self.num_std_for_scale_[c] = 1.0

        # 4) categorical categories (store train cats; unknowns will be ignored)
        for c in self.cat_cols:
            if c in X.columns:
                cats = pd.Index(pd.Series(X[c], dtype="object").dropna().unique())
                # Use a deterministic order:
                self.cat_categories_[c] = pd.Index(sorted(cats.astype(str)))

        # 5) build feature names (without bias)
        self._build_feature_names()
        self.is_fitted_ = True
        return self

    def _build_feature_names(self):
        """Helper method to build feature names"""
        num_names = self.num_med_cols + self.num_mean_cols
        cat_names = []
        for c in self.cat_cols:
            if c in self.cat_categories_:
                cats = self.cat_categories_[c]
                # drop_first=True -> drop the first category
                cats_keep = cats[1:] if self.drop_first and len(cats) > 0 else cats
                cat_names += [f"{c}={val}" for val in cats_keep]
        self.feature_names_ = np.array(num_names + cat_names, dtype=object)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()

        # 1) impute using train stats
        for c in self.num_med_cols:
            if c in X.columns and c in self.medians_:
                X[c] = X[c].fillna(self.medians_[c])
        for c in self.num_mean_cols:
            if c in X.columns and c in self.means_:
                X[c] = X[c].fillna(self.means_[c])

        # 2) scale numeric
        num_all = self.num_med_cols + self.num_mean_cols
        X_num = []
        for c in num_all:
            if c in X.columns and c in self.num_mean_for_scale_:
                mu = self.num_mean_for_scale_[c]
                sd = self.num_std_for_scale_[c]
                X_num.append(((X[c].astype(float) - mu) / sd).to_numpy())
        X_num = np.column_stack(X_num) if X_num else np.empty((len(X), 0))

        # 3) one-hot categorical using TRAIN categories
        X_cat_parts = []
        for c in self.cat_cols:
            if c in X.columns and c in self.cat_categories_:
                cats = self.cat_categories_[c]
                # force to training categories (unknown -> NaN -> all zeros after dummies)
                col = pd.Categorical(X[c].astype("object"), categories=cats)
                dummies = pd.get_dummies(col, prefix=c, prefix_sep='=', dummy_na=False)
                if self.drop_first and dummies.shape[1] > 0:
                    dummies = dummies.iloc[:, 1:]  # drop first category
                X_cat_parts.append(dummies.to_numpy(dtype=float))
        X_cat = np.column_stack(X_cat_parts) if X_cat_parts else np.empty((len(X), 0))

        # 4) concat numeric + categorical
        X_all = np.column_stack([X_num, X_cat]) if X_num.size > 0 or X_cat.size > 0 else np.empty((len(X), 0))

        # 5) add bias as first column
        bias = np.ones((X_all.shape[0], 1), dtype=float)
        X_with_bias = np.hstack([bias, X_all])
        return X_with_bias

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names(self, include_bias=False):
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        if include_bias:
            return np.array(["bias"] + list(self.feature_names_), dtype=object)
        return self.feature_names_.copy()
