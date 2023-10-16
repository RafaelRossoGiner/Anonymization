import os

import numba
import numpy as np
import tqdm

from dataclasses import dataclass
from typing import List, Optional

from itertools import product

from collections import OrderedDict

import Config as C

RootID = -2
MissingInfoID = -1


@numba.njit()
def taxonomy_tree_distance(record1_cats, record2_cats, single_attr_distances):
    total = 0
    for i in range(record1_cats.shape[0]):
        val1 = record1_cats[i]
        val2 = record2_cats[i]
        d = single_attr_distances[(i, val1, val2)]
        total += d
    return total


@numba.njit()
def kmember_distance(rec1c, rec2c, rec1n, rec2n, single_attr_dist, num_attribute_ranges):
    cd = taxonomy_tree_distance(rec1c, rec2c, single_attr_dist)

    nd = np.abs((rec1n - rec2n) / num_attribute_ranges).sum()

    return cd + nd


@numba.njit()
def kmember_dists_to_centroid(clusterCatArray, clusterNumArray, CentroidCatArray, CentroidNumArray, single_attr_dist,
                              num_attribute_ranges):
    n_data_points = clusterCatArray.shape[0]
    distance_to_centroid = np.zeros((n_data_points, 1), dtype=np.float64)

    for i in range(n_data_points):
        cd = taxonomy_tree_distance(clusterCatArray[i, :], CentroidCatArray, single_attr_dist)
        nd = np.abs((clusterNumArray[i, :] - CentroidNumArray) / num_attribute_ranges).sum()
        distance_to_centroid[i] = cd + nd
    return distance_to_centroid


def memoize(func):
    cache = {}

    def memoized(*args):
        cache_key = args[1:]
        cache_key = tuple(k.value if k is not None else None for k in cache_key)

        if cache_key in cache.keys():
            return cache[cache_key]
        else:
            val = func(*args)
            cache[cache_key] = val
            return val

    return memoized


class TreeNode:
    def __init__(self,
                 value: int,
                 children,
                 parent):
        self.value = value
        self.children = children
        self.parent = parent

    def __repr__(self):
        return f"{self.value} {self.children}"

    def height(self) -> int:
        if self.parent is None:
            return 0

        return 1 + self.parent.height()


@dataclass
class Tree:
    # Every tree grows from the root which always has the value '*'
    root: TreeNode
    # Due to building the tree at once we can precalculate the height
    height: int
    labels: dict

    def find_node(self, value: int) -> Optional[TreeNode]:
        # Start the search from the root
        queue = [self.root]

        # Breadth first search over the tree
        cont = 0
        while len(queue) > 0 and cont < 100:
            node = queue.pop()

            if node.value == value:
                return node

            for c in node.children:
                queue.append(c)
            cont += 1

        return None

    def add_node(self, value: int, parent_value: int):
        # Find the parent node and this new node
        parent = self.find_node(parent_value)

        # Check if already present via any and a generator
        # the meaning here is that if any children has this label
        # it means it was added before
        was_already_added = any((c.value == value for c in parent.children))

        # If not added, add it as a new node with no children
        if not was_already_added:
            parent.children.append(TreeNode(value, [], parent))

    def add_path(self, path: List[str]):
        path = path[::-1]

        # This zip operation will create all the child-parent pairs necessary to complete the chain
        # from the initial to the final node of the path.
        for parent_value, value in zip(path, path[1:]):
            self.add_node(value, parent_value)

    @memoize
    def find_matching_parent(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        # If the node does not exist in the tree, then the matching parent is set to be the root
        if node1 is None or node2 is None:
            return self.root

        # If the values are the same, they are the same node
        # so parent for distance calculation purposes is the same one
        # as in paper
        if node1.value == node2.value:
            return node1

        # If the parent is directly above then just return it
        if node1.parent == node2.parent:
            return node1.parent

        # If we reached root, return it
        if node1.parent is None:
            return node1
        if node2.parent is None:
            return node2

        # Examine parent nodes for searching higher in the taxonomy tree
        o1 = self.find_matching_parent(node1, node2.parent)
        o2 = self.find_matching_parent(node1.parent, node2)
        o3 = self.find_matching_parent(node1.parent, node2.parent)

        # Pick the matching parent with largest height
        choices = [o1, o2, o3]
        choices = sorted(choices, key=lambda o: o.height(), reverse=True)

        return choices[0]

    def find_matching_parent_many(self, labels: List[str]) -> TreeNode:
        parents = set()

        for l1, l2 in product(labels, repeat=2):
            mp = self.find_matching_parent(self.find_node(l1),
                                           self.find_node(l2))

            parents = parents | {mp}

        # Best parent is one with minimum height
        parent = min(parents, key=lambda n: n.height())

        return parent

    def all_nodes(self) -> List[TreeNode]:
        output = []
        queue = [self.root]

        while len(queue) > 0:
            node = queue.pop()

            for c in node.children:
                queue.append(c)

            output.append(node)

        return list(set(output))


@dataclass
class TaxonomyForest:
    # Every tree grows from the root which always has the value '*'
    root: TreeNode
    # Due to building the tree at once we can precalculate the height
    trees: dict

    def __init__(self, dataset_df, cat_attr, num_attr, datasetID, yClass):
        self.dataFrame = dataset_df
        self.cat_attr = cat_attr
        self.num_attr = num_attr
        self.datasetID = datasetID
        self.yClass = yClass
        self.progress = None

        self.dataFrame = self.dataFrame.applymap(lambda v: v.strip() if type(v) == str else v)
        missingCatData = self.dataFrame[self.cat_attr] == '?'
        self.dataFrame[self.cat_attr] = self.dataFrame[self.cat_attr].astype('category')
        self.dataFrame[self.cat_attr] = self.dataFrame[self.cat_attr].apply(lambda x: x.cat.codes)

        self.single_attr_distances = numba.typed.Dict()
        self.numerical_attribute_ranges = self.dataFrame.loc[:, self.num_attr].max(axis=0
                                                                                   ) - self.dataFrame.loc[:,
                                                                                       self.num_attr].min(axis=0)

        self.read_all_trees({self.cat_attr[attr]: dict(
            map(lambda i, j: (i, j), dataset_df[self.cat_attr].iloc[:, attr].unique(),
                self.dataFrame[self.cat_attr].iloc[:, attr].unique())) for attr in range(len(self.cat_attr))})

        self.dataFrame[missingCatData] = MissingInfoID

        # Populate the dictionary by going over every attribute.
        self.precalculate_distances()

    @staticmethod
    def read_tree(filepath: str, labels) -> Tree:
        labels['*'] = RootID
        with open(filepath, 'r') as f:
            ls = f.readlines()
            paths = [l.strip().strip('\n').split(';') for l in ls]
            paths = [[labels[lab] for lab in path] for path in paths]
            height = max(map(len, paths)) - 1

            root = TreeNode(RootID, [], None)
            tree = Tree(root, height, labels)

            for path in paths:
                tree.add_path(path)

            return tree

    def read_all_trees(self, catDataFrame):

        self.trees = {}
        TreesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(self.datasetID))
        for attr in self.cat_attr:
            if attr != self.yClass:
                TreePath = os.path.join(TreesDir, C.TAXONOMY_TREE_FILENAME_PREFIX + attr + '.' +
                                        C.TAXONOMY_TREE_EXTENSION)
                self.trees[attr] = self.read_tree(TreePath, catDataFrame[attr])

        self.trees = OrderedDict(self.trees)

    def single_attr_distance(self, attr_name, val1, val2):

        tree = self.trees[attr_name]

        node1 = tree.find_node(val1)
        node2 = tree.find_node(val2)

        mp = tree.find_matching_parent(node1, node2)

        return 1.0 - mp.height() / tree.height

    def precalculate_distances(self):
        # Populate the dictionary by going over every attribute.
        self.progress = tqdm.trange(len(self.cat_attr))
        self.progress.set_description("Initializing Taxonomy Tree operations")
        for attr in self.progress:
            # Get all labels for one tree
            attr = int(attr)
            # labels = list(self.dataFrame.loc[:, self.cat_attr].iloc[:, attr].unique())
            labels = [l.value for l in self.trees[self.cat_attr[attr]].all_nodes()]

            # For each pair of labels calculate the distance
            # Product() will create an iterable list will all possible pair permutations of the labels
            for l1, l2 in product(labels, repeat=2):
                self.single_attr_distances[(attr, int(l1), int(l2))] = self.single_attr_distance(self.cat_attr[attr],
                                                                                                 l1, l2)

            # Handle missing values cases
            for lab in labels:
                lab = int(lab)
                self.single_attr_distances[(attr, int(MissingInfoID), lab)] = np.float64(1.0)
                self.single_attr_distances[(attr, lab, int(MissingInfoID))] = np.float64(1.0)

            # Distance between two missing values is 1.0, the maximal possible distance
            self.single_attr_distances[(attr, int(MissingInfoID), int(MissingInfoID))] = np.float64(1.0)

    def information_loss_cluster(self, cl_id: int) -> float:
        data = self.dataFrame[self.dataFrame.equivalence_class == cl_id].copy()
        n_points = data.shape[0]

        loss = 0

        # for categorical features
        # for all pairs of present labels find parents
        # of all parents find the one with lowest height
        # as the worst possible case
        # and have that as the parent for distance calculation
        for attr in range(len(self.cat_attr)):
            labels = data.iloc[:, attr].unique()
            dists = []
            for l1, l2 in product(labels, repeat=2):
                dists.append(self.single_attr_distances[(attr, l1, l2)])

            loss += (min(dists) if len(dists) > 0 else 0.0)

        # for numeric features find the ratio of range in cluster over
        # range in full dataset
        for fid in range(len(self.cat_attr), data.shape[1] - 1):
            v = data.iloc[:, fid].max() - data.iloc[:, fid].min()
            if np.isnan(v):
                v = 0.0
            v = v / self.numerical_attribute_ranges[fid - len(self.cat_attr)]
            loss += v

        loss *= n_points

        return loss

    def information_loss(self):
        clusterLabels = self.dataFrame['equivalence_class'].unique()

        loss_vector = {}
        total_loss = 0

        for cl_id in range(len(clusterLabels)):
            loss_vector[cl_id] = self.information_loss_cluster(clusterLabels[cl_id])
            total_loss += loss_vector[cl_id]

        return loss_vector, total_loss

    def oka_centroid(self, cluster):
        numeric_attributes = cluster.iloc[:, len(self.cat_attr):].mean().values.tolist()

        # Matching parent for all pairs of leaves in this cluster
        categorical_attributes = []

        for fid in range(len(self.cat_attr)):
            labels = cluster.iloc[:, fid].unique()
            attr = self.cat_attr[fid]
            tree = self.trees[attr]

            # Find parent with lowest distance to the label
            parent = tree.find_matching_parent_many(labels)

            categorical_attributes.append(parent.value)

        return np.array(categorical_attributes + numeric_attributes, dtype=np.object)

    def adjustment_stage(self, desired_k=50):
        adjusted_cluster = self.dataFrame['equivalence_class'].copy()

        record_index = np.arange(0, self.dataFrame.shape[0])

        self.dataFrame.set_index(record_index, inplace=True)

        clusterIDs = self.dataFrame['equivalence_class'].unique()
        n_clusters = len(clusterIDs)

        # Generate a dictionary containing the clustered data points, separated according to the SOM results.
        clusters = {
            cl_id: self.dataFrame[adjusted_cluster == clusterIDs[cl_id]].drop('equivalence_class', axis=1).copy()
            for cl_id in range(n_clusters)}

        # Generate centroids
        centroids = {int(cl_id): self.oka_centroid(cluster)
                     for cl_id, cluster in clusters.items()}

        # Collect records far away from centroids from big clusters
        self.progress = tqdm.tqdm(clusters.items(), total=n_clusters)
        self.progress.set_description("Collecting far away records from big clusters.")
        for cl_id, cluster_data in self.progress:
            if cluster_data.shape[0] <= desired_k:
                continue

            # Calculate distance between the centroid and all elements of the cluster
            cluster_data['dist_to_centroid'] = kmember_dists_to_centroid(cluster_data[self.cat_attr].values,
                                                                         cluster_data[self.num_attr].values,
                                                                         centroids[cl_id][:len(self.cat_attr)].astype(
                                                                             int),
                                                                         centroids[cl_id][len(self.cat_attr):].astype(
                                                                             np.float64),
                                                                         self.single_attr_distances,
                                                                         self.numerical_attribute_ranges.values)

            # Order so the first element corresponds to the furthest element from the centroid
            cluster_data.sort_values(by='dist_to_centroid', ascending=False, inplace=True)
            cluster_data.drop('dist_to_centroid', axis=1, inplace=True)

            for i in range(0, cluster_data.shape[0] - desired_k):
                record = cluster_data.iloc[i, :]

                index = record.__dict__['_name']

                adjusted_cluster[index] = -1

        # Distribute records to clusters where they are needed
        R = self.dataFrame[adjusted_cluster == -1]
        self.progress = tqdm.trange(R.shape[0])
        self.progress.set_description("Distributing records to clusters")
        for i in self.progress:
            record = R.iloc[i, :]
            min_d = 10000000
            best_cluster = -1
            for cluster_id, cluster in clusters.items():
                if (adjusted_cluster == cluster_id).sum() > desired_k:
                    continue
                d = kmember_distance(record[self.cat_attr].values,
                                     centroids[cluster_id][:len(self.cat_attr)].astype(int),
                                     record[self.num_attr].values,
                                     centroids[cluster_id][len(self.cat_attr):].astype(np.float64),
                                     self.single_attr_distances,
                                     self.numerical_attribute_ranges.values)
                if d <= min_d:
                    min_d = d
                    best_cluster = cluster_id

            index = record.__dict__['_name']
            adjusted_cluster[index] = best_cluster

        # Recalculate remaining clusters
        clusterIDs = adjusted_cluster[adjusted_cluster != -1].unique()
        n_clusters = len(clusterIDs)
        clusters = {
            cl_id: self.dataFrame[adjusted_cluster == clusterIDs[cl_id]].drop('equivalence_class', axis=1).copy()
            for cl_id in range(n_clusters)}

        # centroids = {int(cl_id): self.oka_centroid(cluster) for cl_id, cluster in clusters.items()}

        # Collect any record in isolated clusters
        self.progress = tqdm.tqdm(clusters.copy().items(), total=n_clusters)
        self.progress.set_description("Collecting spare records from isolated clusters.")
        for cl_id, cluster_data in self.progress:
            if cluster_data.shape[0] >= desired_k:
                continue

            for i in range(0, cluster_data.shape[0]):
                record = cluster_data.iloc[i, :]
                index = record.__dict__['_name']
                adjusted_cluster[index] = -1
            clusters.pop(cl_id)
            centroids.pop(cl_id)

        # Recalculate remaining clusters
        clusterIDs = adjusted_cluster[adjusted_cluster != -1].unique()
        n_clusters = len(clusterIDs)
        clusters = {
            cl_id: self.dataFrame[adjusted_cluster == clusterIDs[cl_id]].drop('equivalence_class', axis=1).copy()
            for cl_id in range(n_clusters)}

        centroids = {int(cl_id): self.oka_centroid(cluster) for cl_id, cluster in clusters.items()}

        # Distribute remaining data to their closest cluster
        R = self.dataFrame[adjusted_cluster == -1]
        self.progress = tqdm.trange(R.shape[0])
        self.progress.set_description("Distributing remaining records")
        for i in self.progress:
            record = R.iloc[i, :]
            min_d = 10000000
            best_cluster = -1
            for cluster_id, cluster in clusters.items():
                d = kmember_distance(record[self.cat_attr].values,
                                     centroids[cluster_id][:len(self.cat_attr)].astype(int),
                                     record[self.num_attr].values,
                                     centroids[cluster_id][len(self.cat_attr):].astype(np.float64),
                                     self.single_attr_distances,
                                     self.numerical_attribute_ranges.values)
                if d <= min_d:
                    min_d = d
                    best_cluster = clusterIDs[cluster_id]

            index = record.__dict__['_name']
            adjusted_cluster[index] = best_cluster

        self.dataFrame['equivalence_class'] = adjusted_cluster

        clusterIDs = self.dataFrame['equivalence_class'].unique()
        n_clusters = len(clusterIDs)

        # Recalculate clusters
        clusters = {
            cl_id: self.dataFrame[self.dataFrame.equivalence_class == clusterIDs[cl_id]].drop('equivalence_class',
                                                                                              axis=1).copy()
            for cl_id in range(n_clusters)}

        # Recalculate centroids
        centroids = {int(cl_id): self.oka_centroid(cluster)
                     for cl_id, cluster in clusters.items()}

        inverseLabels = {attr: {labId: lab for lab, labId in self.trees[attr].labels.items()} for attr in self.cat_attr}

        for key, centroid in centroids.items():
            # Translate back all ids to string
            for i in range(len(self.cat_attr)):
                centroid[i] = inverseLabels[self.cat_attr[i]][centroid[i]]

        return self.dataFrame, centroids  # , clusters, centroids
