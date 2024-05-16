from __future__ import annotations
import math

import numpy
import pandas as pd
from ID3.DecisonTree import Leaf, Question, DecisionNode, class_counts, unique_vals
from ID3.utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        _entropy = 0.0

        samples = len(rows)
        probabilities = [value / samples for key, value in counts.items()]
        _entropy = -np.sum([num * np.log2(num) for num in probabilities])
        return _entropy

    def info_gain(self, left, left_labels, right, right_labels, current_info_gain=None):
        """
        Calculate the information gain, as the current_info_gain of the starting node, minus the weighted entropy of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_info_gain: the current info_gain of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        total_samples = len(left) + len(right)
        left_weight = len(left) / total_samples
        right_weight = len(right) / total_samples
        info_gain_value = current_info_gain - right_weight * self.entropy(right, right_labels) - \
                          left_weight * self.entropy(left, left_labels)

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = 0, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # Assuming rows is a numpy array and labels is a numpy array
        true_indices = [i for i, row in enumerate(rows) if question.match(row)]
        false_indices = [i for i, row in enumerate(rows) if not question.match(row)]

        if len(true_indices):
            true_rows = rows[np.array(true_indices)]
            true_labels = labels[np.array(true_indices)]
        if len(false_indices):
            false_rows = rows[np.array(false_indices)]
            false_labels = labels[np.array(false_indices)]

        # right is true
        gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        for feature in range(len(rows[0])):
            for value in unique_vals(rows, feature):
                question = Question(self.label_names[feature], feature, value)  # todo: feature name
                gain, true_rows, true_labels, false_rows, false_labels = \
                    self.partition(rows, labels, question, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question, best_false_rows, best_false_labels, best_true_rows, best_true_labels = \
                        gain, question, false_rows, false_labels, true_rows, true_labels

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children

        if isinstance(rows[0], np.float64) or len(rows) <= self.min_for_pruning:
            print("1 sample")
            print(rows.shape)
            return Leaf(rows, labels)

        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
            self.find_best_split(rows, labels)

        if best_gain <= 0:
            print("no gain")
            print(rows.shape)
            return Leaf(rows, labels)

        true_branch, false_branch = self.build_tree(best_true_rows, best_true_labels), \
                                    self.build_tree(best_false_rows, best_false_labels)

        print(best_question.column, best_question.value)
        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root
        self.tree_root = self.build_tree(x_train, y_train)

    def predict_sample(self, row, node: DecisionNode | Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        while isinstance(node, DecisionNode):
            if node.question.match(row):
                node = node.true_branch
            else:
                node = node.false_branch

        prediction = max(node.predictions, key=node.predictions.get)

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = [self.predict_sample(row) for row in rows]

        return y_pred
