import numpy as np
from scipy.optimize import minimize
import pandas as pd
from numba import jit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.cluster import k_means


def add_ones_col(X: np.ndarray):
    h, w = X.shape
    _X = np.zeros((h, w + 1))
    _X[:, 0] = 1
    _X[:, 1:] = X.copy()
    return _X


def cost_val(D):
    if D.size == 0:
        return 0
    else:
        return (np.linalg.norm(D - D.mean(axis=0), axis=1) ** 2).sum()


def opt_node(Theta, *args):
    node_data, node_values = args
    below_hyperplane = node_data @ Theta < 0
    above_hyperplane = (1 - below_hyperplane).astype(bool)
    return cost_val(node_values[below_hyperplane].copy()) + cost_val(node_values[above_hyperplane].copy())


class NodeBase(object):
    def __init__(self, point_set, value_set, depth=0, node_id=0, split_method='naive', parent=None):

        self.ps = point_set
        self.vs = value_set
        if self.ps.shape[0] > 1:
            self.offset = self.ps.mean(axis=0)
        else:
            self.offset = self.ps

        self.parent = parent
        self.child_right = None
        self.child_left = None
        self.size = self.ps.shape[0]
        if self.vs.shape[0] > 1:
            self.value = self.vs.mean(axis=0)
        else:
            self.value = self.vs
        self.is_leaf = True
        self.svm_func = None
        self.method = split_method
        self.Theta = None
        self.depth = depth
        self.nid = node_id

    # def get_var(self):
    #     return cost_val(self.vs) / self.size

    def next_node(self, x):

        if self.method == 'naive':
            return self.child_right.nid if (x * self.Theta).sum() < 0 else self.child_left.nid
        elif self.method == 'svmlike':
            nnid = self.child_right.nid if self.svm_func(x) == 0 else self.child_left.nid
            return nnid

    def copy(self):
        _n = NodeBase(self.ps, self.vs, self.depth, self.nid, split_method=self.method)
        _n.is_leaf = self.is_leaf
        if not self.is_leaf:
            _n.Theta = self.Theta
            _n.child_right = self.child_right.copy()
            _n.child_left = self.child_left.copy()
            try:
                _n.svm_func = lambda x: self.svm_func(x)
            except AttributeError as e:
                print(e)
                _n.svm_func = None
        return _n

    def _divide(self):
        """
        Find indices of all points 'below' hyperplane and indices of points "above" hyperplane
        :return: (List[int], List[int])
        """
        below_ind, above_ind = None, None
        if self.method == "naive":
            below_ind = add_ones_col(self.ps - self.offset) @ self.Theta < 0
            above_ind = (1 - below_ind).astype(bool)

        elif self.method == "svmlike":
            below_ind = self.svm_func(add_ones_col(self.ps - self.offset)) == 0
            below_ind = below_ind.reshape(below_ind.size)
            above_ind = (1 - below_ind).astype(bool)
        # if one of the sides is empty - you have reached a leaf
        if below_ind.sum() == 0 or above_ind.sum() == 0:

            self.is_leaf = True
        else:
            self.is_leaf = False
        return below_ind, above_ind

    def _get_hyper_plane_coefficients(self):
        """
        Find coefficients theta_0,...,theta_n which minimizing variance according to the paper
        :return: np.ndarray: Theta
        """
        cent_ps = add_ones_col(self.ps - self.offset)

        opt_res = minimize(opt_node, x0=np.zeros(cent_ps.shape[1]), args=(cent_ps, self.vs.copy()),
                           method='Nelder-Mead')
        return opt_res.x

    @jit
    def split(self, rid, lid):
        """
        splits the node into two : finds an optimal hyperplane by which the data points are best divided by and
         adds left and right children with subsets of data 'below' and 'above' the hyperplane found
        :return:None
        """
        if self.size <= 1:
            # just in case someone tries to split a single sample
            self.is_leaf = True
        else:
            self.is_leaf = False
            if self.method == 'naive':
                self.Theta = self._get_hyper_plane_coefficients()  # Find hyperplane equation
            elif self.method == 'svmlike':
                retcode = self._get_hyper_plane_coefficients_svm()  # Find hyperplane equation
            if self.is_leaf:
                # remove Theta so no confusions will arise
                self.Theta = None

            else:
                # add children
                below_ind, above_ind = self._divide()  # divide dataset to below and above
                self.child_right = NodeBase(self.ps[below_ind].copy() - self.offset, self.vs[below_ind].copy(),
                                            depth=self.depth + 1, node_id=rid, split_method=self.method, parent=self)
                self.child_left = NodeBase(self.ps[above_ind].copy() - self.offset, self.vs[above_ind].copy(),
                                           depth=self.depth + 1, node_id=lid, split_method=self.method, parent=self)

    def _get_hyper_plane_coefficients_svm(self):
        centroids, clusters, inertia = k_means(self.vs, n_clusters=2 )
        if (clusters == 0).sum() == 0 or (clusters == 1).sum() == 0:
            self.is_leaf = True
            return -1
        lda = SVC(kernel='linear')
        try:
            lda = lda.fit(add_ones_col(self.ps - self.offset), clusters)
            self.svm_func = lda.predict
            return 0
        except ZeroDivisionError as e:
            self.is_leaf = True
            return -1


class TreeBase(object):
    def __init__(self, min_samples_split=2, min_samples_leaf=1, split_method='naive'):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.method = split_method

        self.dict_tree = {'Thetas': None,
                          'Values': None,
                          'Nodes': {}}

    # noinspection SpellCheckingInspection
    def _dictinit(self, ps, vs):
        self.dict_tree['Thetas'] = pd.DataFrame(columns=range(ps.shape[1] + 1))
        self.dict_tree['Values'] = pd.DataFrame(columns=range(vs.shape[1]))

    def fit(self, point_set, value_set):
        _vs = value_set.copy()
        if _vs.shape[0] == _vs.size:
            _vs = _vs.reshape((_vs.size, 1))
        self._dictinit(point_set.copy(), _vs)
        self.root = NodeBase(point_set.copy(), _vs, split_method=self.method)
        # build the DT
        untraversed_node_list = [self.root]
        it = 0
        while len(untraversed_node_list) > 0:
            _node = untraversed_node_list.pop()

            if _node.size < self.min_samples_split:
                _node.is_leaf = True
            else:
                if _node is None:
                    print("None")
                _node.split(it, it + 1)
                if not _node.is_leaf:
                    untraversed_node_list = [_node.child_left, _node.child_right] + untraversed_node_list
                    self.dict_tree['Thetas'].loc[_node.nid, :] = _node.Theta
                    # self.dict_tree['']
            self.dict_tree['Values'].loc[_node.nid, :] = _node.value
            self.dict_tree['Nodes'][_node.nid] = _node
            it += 2
        return self

    @jit
    def _predict_point(self, x):
        _nid = 0
        # _n = self.root.copy()
        _x = np.ones((1, x.size + 1))
        _x[0, 1:] = x
        while not self.dict_tree['Nodes'][_nid].is_leaf:
            _x[0, 1:] = _x[0, 1:] - self.dict_tree['Nodes'][_nid].offset
            _nid = self.dict_tree['Nodes'][_nid].next_node(_x)
            # _n = self.dict_tree['Nodes'][_next_nid]
        return self.dict_tree['Nodes'][_nid].value

    @jit
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.root.vs.shape[1]))

        for i in range(X.shape[0]):
            _x = X[i].copy()
            y_pred[i] = self._predict_point(_x)
        return y_pred
