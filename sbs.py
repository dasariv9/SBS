import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

class SBS:
    """
    Sequential Backward Selection (SBS) algorithm for feature selection.
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Initial full feature set
        n_features = X_train.shape[1]
        self.indices_ = tuple(range(n_features))
        self.subsets_ = [self.indices_]
        self.scores_ = []

        # Initial score with all features
        initial_score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_.append(initial_score)

        # Backward elimination loop
        while n_features > self.k_features:
            scores = []
            candidates = []

            # Try removing one feature at a time
            for subset in combinations(self.indices_, r=n_features - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, subset)
                scores.append(score)
                candidates.append(subset)

            # Select best-performing subset
            best_idx = np.argmax(scores)
            self.indices_ = candidates[best_idx]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best_idx])
            n_features -= 1

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, feature_indices):
        self.estimator.fit(X_train[:, feature_indices], y_train)
        y_pred = self.estimator.predict(X_test[:, feature_indices])
        return self.scoring(y_test, y_pred)
