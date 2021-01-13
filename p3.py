import numpy as np
import math
import csv


def read_data(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        return list(next(datareader)), list(datareader)


class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

    def __str__(self):
        return self.attribute


def entropy(S):
    items = np.unique(S)
    if items.size == 1:
        return 0
    sums = 0
    for x in items:
        p = sum(S == x) / S.size
        sums += p * math.log(p, 2)
    return -sums


def gain_ratio(data, col):
    features, dict = subtables(data, col, delete=False)

    entropies = np.zeros(len(features))
    intrinsic = np.zeros(len(features))

    for i in range(len(features)):
        # fraction of examples that end up on this sub-node
        frac = len(dict[features[i]]) / len(data)

        entropies[i] = frac * entropy(dict[features[i]][:, -1])
        intrinsic[i] = frac * math.log(frac, 2)

    total_entropy = entropy(data[:, -1]) - np.sum(entropies)
    iv = -np.sum(intrinsic)
    return total_entropy / iv


def create_node(data, cols):
    features = np.unique(data[:, -1])
    if len(features) == 1:
        node = Node("")
        node.answer = features[0]
        return node

    gains = np.zeros(len(cols) - 1)
    for i in range(len(gains)):
        gains[i] = gain_ratio(data, i)

    max_idx = np.argmax(gains)
    node = Node(cols[max_idx])
    cols = np.delete(cols, max_idx)

    features, dict = subtables(data, max_idx, delete=True)
    for feature in features:
        child = create_node(dict[feature], cols)
        node.children.append((feature, child))

    return node


def subtables(data, col, delete):
    dict = {}
    features = np.unique(data[:, col])
    for value in features:
        dict[value] = data[data[:, col] == value]
        if delete:
            dict[value] = np.delete(dict[value], col, 1)
    return features, dict


def print_tree(node, level):
    if node.answer != "":
        print(" " * level + node.answer)
        return
    print(" " * level + node.attribute)
    for value, n in node.children:
        print(" " * (level + 1) + value)
        print_tree(n, level + 2)


def empty(size):
    s = ""
    for x in range(size):
        s += " "
    return s


if __name__ == "__main__":
    cols, traindata = read_data("p3.csv")
    data = np.array(traindata)
    node = create_node(data, cols)
    print_tree(node, 0)
