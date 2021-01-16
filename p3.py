import numpy as np
import math
import csv


class Node:
    def __init__(self, feature):
        self.cls = None
        self.feature = feature
        self.children = []

    def print_tree(self, level=0):
        if self.cls:
            print(" " * level + self.cls)
            return

        print(" " * level + self.feature)

        for value, child in self.children:
            print(" " * (level + 1) + value)
            child.print_tree(level + 2)


def build(data, features):
    classes = np.unique(data[:, -1])
    if len(classes) == 1:
        node = Node("")
        node.cls = classes[0]
        return node

    gains = np.zeros(len(features) - 1)
    for feature in range(len(gains)):
        gains[feature] = gain_ratio(data, feature)

    best_feature = np.argmax(gains)
    node = Node(features[best_feature])
    features = np.delete(features, best_feature)

    values, tables = subtables(data, best_feature, delete=True)
    for value in values:
        child = build(tables[value], features)
        node.children.append((value, child))

    return node


def gain_ratio(data, feature):
    values, tables = subtables(data, feature, delete=False)

    entropies = np.zeros(len(values))
    intrinsic = np.zeros(len(values))

    for i in range(len(values)):
        # fraction of examples that end up on this sub-node
        frac = len(tables[values[i]]) / len(data)

        entropies[i] = frac * entropy(tables[values[i]][:, -1])
        intrinsic[i] = frac * math.log(frac, 2)

    total_entropy = entropy(data[:, -1]) - np.sum(entropies)
    iv = -np.sum(intrinsic)
    return total_entropy / iv


def entropy(S):
    items = np.unique(S)
    if items.size == 1:
        return 0
    sums = 0
    for x in items:
        p = sum(S == x) / S.size
        sums += p * math.log(p, 2)
    return -sums


def subtables(data, feature, delete):
    values = np.unique(data[:, feature])
    tables = {}

    for value in values:
        tables[value] = data[data[:, feature] == value]

        if delete:
            tables[value] = np.delete(tables[value], feature, 1)

    return values, tables


def read_data(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        return list(next(datareader)), np.array(list(datareader))


columns, data = read_data("p3.csv")
node = build(data, columns)
node.print_tree()
