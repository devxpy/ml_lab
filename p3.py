import csv

import numpy as np


class Node:
    def __init__(self, attr):
        self.attr = attr
        self.children = []

    def display(self, level=0):
        print(" " * level + self.attr)

        for val, node in self.children:
            print(" " * (level + 1) + val)
            node.display(level + 2)


def subtables(D, fi, delete=False):
    tables = {}
    for xi in np.unique(D[:, fi]):
        tables[xi] = D[D[:, fi] == xi]
        if delete:
            tables[xi] = np.delete(tables[xi], fi, axis=1)
    return tables


def gain(D, fi):
    ig = iv = 0

    tables = subtables(D, fi)
    for table in tables.values():
        p = len(table) / len(D)

        ig += p * entropy(table[:, -1])
        iv += p * np.log2(p)

    ig = entropy(D[:, -1]) - ig
    return ig / -iv


def entropy(S):
    E = 0
    for xi in np.unique(S):
        p = np.sum(S == xi) / len(S)
        E += p * np.log2(p)
    return -E


def create_node(D, features):
    classes = np.unique(D[:, -1])
    if len(classes) == 1:
        return Node(classes[0])

    gains = [gain(D, fi) for fi in range(len(features))]
    fi = np.argmax(gains)
    root = Node(features[fi])

    features = np.delete(features, fi)

    tables = subtables(D, fi, delete=True)
    for val, table in tables.items():
        child = create_node(table, features)
        root.children.append((val, child))

    return root


def main():
    with open("p3.csv") as f:
        reader = csv.reader(f)
        features = np.array(next(reader)[:-1])
        node = create_node(np.array(list(reader)), features)
        node.display()


if __name__ == "__main__":
    main()
