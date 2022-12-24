import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import time
from functools import partial


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        # self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        # self.trees_parameters = trees_parameters
        self.n_objects = None
        self.n_features = None
        self.estimators = [DecisionTreeRegressor(max_depth=max_depth,
                                                 **trees_parameters)
                           for _ in range(self.n_estimators)]
        self.loss = None
        self.f_arr = []
        self.time = []

    def fit(self, X, y, X_val=None, y_val=None, is_loss_all=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        flag_val = True
        if is_loss_all:
            self.loss = []
        if X_val is None or y_val is None:
            flag_val = False

        self.n_objects = X.shape[0]
        self.n_features = X.shape[-1]

        if self.feature_subsample_size is None:
            self.feature_subsample_size = self.n_features // 3
        if flag_val:
            pred = np.zeros((len(self.estimators), len(y_val)))
        self.f_arr = []

        for i, tree in enumerate(self.estimators):
            if is_loss_all:
                start = time.time()
            s = np.random.randint(self.n_objects, size=self.n_objects)
            f = np.random.choice(self.n_features, replace=False,
                                 size=self.feature_subsample_size)
            self.f_arr.append(f)
            X_boots = X[s, :][:, f]
            tree.fit(X_boots, y[s])
            if flag_val:
                y_pr = tree.predict(X_val[:, f])
                pred[i] = y_pr
                if is_loss_all:
                    y_pred_i = np.mean(pred[:i+1], axis=0)
                    self.time.append(time.time() - start)
                    #   print(y_val.shape, y_pred_i.shape)
                    self.loss.append(mean_squared_error(
                        y_val, y_pred_i, squared=False))
        if flag_val:
            if is_loss_all:
                # for i in range(1, self.n_estimators+1):
                #     y_pred_i = np.mean(pred[:i], axis=0)
                #     loss_all.append(mean_squared_error(
                #         y_val, y_pred_i, squared=False))
                pass
            else:
                y_pred = np.mean(pred, axis=0)
                self.loss = mean_squared_error(
                    y_val, y_pred, squared=False)

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        predictions = []
        for i, tree in enumerate(self.estimators):
            predictions.append(tree.predict(X[:, self.f_arr[i]]))
        return np.mean(predictions, axis=0)


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5,
            feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.

        """

        self.n_estimators = n_estimators
        # self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        # self.trees_parameters = trees_parameters
        self.n_features = None
        self.estimators = [DecisionTreeRegressor(
            max_depth=max_depth, **trees_parameters) for _ in range(self.n_estimators)]
        self.loss = []
        self.f_arr = []
        self.time = []
        self.lr = learning_rate
        self.F0 = None

    def loss_calc(self, alpha, y_true, b, f):

        loss = (1/len(y_true)) * 0.5*np.sum(np.square(y_true-(f+alpha*b)))

        return loss

    def fit(self, X, y, X_val=None, y_val=None, is_loss_all=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        flag_val = True
        if is_loss_all:
            self.loss = []
        if X_val is None or y_val is None:
            flag_val = False

        # self.n_objects = X.shape[0]
        self.n_features = X.shape[-1]

        if self.feature_subsample_size is None:
            self.feature_subsample_size = self.n_features // 3
        if flag_val:
            pred = np.zeros((len(self.estimators), len(y_val)))
        self.f_arr = []
        self.F0 = np.mean(y)
        Ft = self.F0

        for i, tree in enumerate(self.estimators):
            if is_loss_all:
                start = time.time()
            f = np.random.choice(self.n_features, replace=False,
                                 size=self.feature_subsample_size)
            self.f_arr.append(f)
            X_boots = X[:, f]
            tree.fit(X_boots, y-Ft)
            b = tree.predict(X[:, f])
            func = partial(self.loss_calc, y_true=y, b=b, f=Ft)
            alpha = minimize_scalar(func, bounds=(0, float('inf'))).x
            Ft += self.lr*alpha*b

            if flag_val:
                y_pr = tree.predict(X_val[:, f])
                pred[i] = y_pr
                if is_loss_all:
                    y_pred_i = self.F0 + self.lr*np.sum(pred[:i+1], axis=0)
                    self.time.append(time.time() - start)
                    #   print(y_val.shape, y_pred_i.shape)
                    self.loss.append(mean_squared_error(
                        y_val, y_pred_i, squared=False))
        if flag_val:
            if is_loss_all:
                # for i in range(1, self.n_estimators+1):
                #     y_pred_i = np.mean(pred[:i], axis=0)
                #     loss_all.append(mean_squared_error(
                #         y_val, y_pred_i, squared=False))
                pass
            else:
                y_pred = self.F0 + self.lr*np.sum(pred, axis=0)
                self.loss = mean_squared_error(
                    y_val, y_pred, squared=False)

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = []
        for i, tree in enumerate(self.estimators):
            predictions.append(tree.predict(X[:, self.f_arr[i]]))
        return self.F0 + self.lr*np.sum(predictions, axis=0)
