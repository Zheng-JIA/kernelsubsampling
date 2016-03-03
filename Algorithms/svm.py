import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from time import time
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import accuracy_score
class kernelsvm():
    def __init__(self, theta0, alpha, loss_metric):
        self.theta0 = theta0
        self.alpha = alpha
        self.loss_metric = loss_metric
    def fit(self, X, y, idx_SR):
        n_SR = len(idx_SR)
        self.feature_map_nystroem = General_Nystroem(kernel='rbf', gamma=self.theta0, n_components=n_SR)
        X_features = self.feature_map_nystroem.fit_transform(X,idx_SR)
        print("fitting SGD")
        self.clf = SGDClassifier(loss=self.loss_metric,alpha=self.alpha)
        self.clf.fit(X_features, y)
        print("fitting SGD finished")
    def predict(self, X):
        print("Predicting")
        X_transform = self.feature_map_nystroem.transform(X)
        return self.clf.predict(X_transform), X_transform
    def decision_function(self, X):
        # X should be the transformed input!
        return self.clf.decision_function(X)
    def err_rate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        err_rate = 1.0-acc
        return err_rate
    def get_params(self):
        return self.clf.get_params()
                                          
class General_Nystroem(BaseEstimator, TransformerMixin):
    """Approximate a kernel map using a subset of the training data.
    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.
    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.
    n_components : int
        Number of features to construct.
        How many data points will be used to construct the mapping.
    gamma : float, default=None
        Gamma parameter for the RBF, polynomial, exponential chi2 and
        sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.
    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Subset of training points used to construct the feature map.
    component_indices_ : array, shape (n_components)
        Indices of ``components_`` in the training set.
    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.
    References
    ----------
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001
    * T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
      "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
      Comparison",
      Advances in Neural Information Processing Systems 2012
    See also
    --------
    RBFSampler : An approximation to the RBF kernel using random Fourier
                 features.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """
    def __init__(self, kernel="rbf", gamma=None, coef0=1, degree=3,
                 kernel_params=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        
    def fit(self, X, idx_SR, y=None):
        """Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Training data.
        """
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        
        #basis_inds = inds[:n_components] is replaced by the following assignment
        basis_inds = idx_SR
        basis = X[basis_inds]
        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        filter_params=True,
                                        **self._get_kernel_params())
        # sqrt of kernel matrix on basis vectors
        print("Decomposing k_SR")
        U, S, V = svd(basis_kernel)
        print("Decomposing finished")
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)
        self.components_ = basis
        print("normalization finished")
        # store other useful parameters
        self.basis_kernel = basis_kernel
        self.U = U
        self.S = S
        self.V = V
        return self
        
    def transform(self, X):
        """Apply feature map to X.
        Computes an approximate feature map using the kernel
        between some training points and X.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Data to transform.
        Returns
        -------
        X_transformed : array, shape=(n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, 'components_')
        X = check_array(X, accept_sparse='csr')
        kernel_params = self._get_kernel_params()
        print("Transform input")
        print("Input size is %d "%len(X)," by %d "%len(self.components_))
        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    filter_params=True,
                                    **kernel_params)
        print("Transform input finished")
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel):
            params['gamma'] = self.gamma
            params['degree'] = self.degree
            params['coef0'] = self.coef0

        return params
