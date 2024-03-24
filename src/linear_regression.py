import numpy as np


class MyLinearRegression:
    # region Constructor

    def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
        # region Summary
        """
        This class implements Linear Regression models
        :param regularization: None for no regularization | "l1" for Lasso Regression | "l2" for Ridge Regression
        :param lam: Lambda parameter for regularization in case of Lasso and Ridge
        :param learning_rate: Learning rate for Gradient Descent algorithm (is used in case of Lasso)
        :param tol: Tolerance level for weight change in Gradient Descent
        """
        # endregion Summary

        self.regularization = regularization
        self.lam = lam
        self.learning_rate = learning_rate
        self.tol = tol
        self.weights = None

    # endregion Constructor

    # region Functions

    def fit(self, X, y):
        # region Summary
        """

        :param X: Dataset
        :param y: Target
        :return: Trained model
        """
        # endregion Summary

        X = np.array(X)

        # First, insert a column with all 1s in the beginning. Hint: use the function np.insert
        X = np.insert(X, 0, 1, axis=1)

        # The case when no regularization is applied
        if self.regularization is None:
            # region Variant N

            self.weights = np.linalg.pinv(X.T @ X) @ (X.T @ y)

            # endregion Variant N

            # region Variant A

            # y = np.array(y)
            # self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            # print(self.weights.shape)

            # endregion Variant A

        # In case of Lasso Regression, the Gradient Descent is used to find the optimal combination of weights that
        # minimizes the objective function in this case (slide 37)
        elif self.regularization == "l1":
            # Initialize random weights, for example, normally distributed

            # region Variant N

            self.weights = np.random.randn(X.shape[1])

            # endregion Variant N

            # region Variant A

            # y = np.array(y).reshape((-1, 1))
            # self.weights = np.random.randn(X.shape[1], 1)

            # endregion Variant A

            converged = False

            # The loss values can be stored to see how fast the algorithm converges
            self.loss = []

            # Counter of algorithm steps
            i = 0
            while not converged:
                i += 1

                # Calculate the predictions in case of the weights in this stage
                y_pred = X @ self.weights

                # region Variant N

                # Calculate the MSE (loss) for the predictions obtained above
                self.loss.append(np.mean((y - y_pred) ** 2))

                # Calculate the gradient of the objective function with respect to w for the second component
                # \sum|w_i| use np.sign(w_i) as it's derivative
                grad = -2 * X.T @ (y - y_pred) + self.lam * np.sign(self.weights)

                # endregion Variant N

                # region Variant A

                # self.loss.append((1 / (2 * len(X))) * np.sum((y_pred - y) ** 2))
                # grad = (1 / len(X)) * X.T @ (X @ self.weights - y) + self.lam * np.sign(self.weights)

                # endregion Variant A

                new_weights = self.weights - self.learning_rate * grad

                # region Variant N

                # Check whether the weights have changed a lot after this iteration. Compute the norm of difference
                # between old and new weights and compare with the pre-defined tolerance level. If the norm
                # is smaller than the tolerance level, then the algorithm is considered to be convergent
                converged = np.linalg.norm(self.weights - new_weights) < self.tol

                # endregion Variant N

                # region Variant A

                # converged = np.sqrt(np.sum((self.weights - new_weights) ** 2)) <= self.tol

                # endregion Variant A

                self.weights = new_weights

                # region Variant A

                # if i > 15000:
                #     break

                # endregion Variant A

            print(f'Converged in {i} steps')

        # The case of Ridge Regression
        elif self.regularization == "l2":
            # region Variant N

            I = np.identity(X.shape[0])
            self.weights = np.linalg.pinv(X.T @ X + self.lam * I) @ (X.T @ y)

            # endregion Variant N

            # region Variant A

            # self.weights = np.linalg.inv(X.T.dot(X) + self.lam * np.identity(len(X))).dot(X.T).dot(y)

            # endregion Variant A

    def predict(self, X):
        # region Summary
        """

        :param X: Dataset
        :return: Prediction
        """
        # endregion Summary

        X = np.array(X)

        # Add the feature of 1s in the beginning
        X = np.insert(X, 0, 1, axis=1)

        # Predict using the obtained weights
        return X @ self.weights

    # endregion Functions
