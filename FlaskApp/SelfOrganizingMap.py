import datetime
import math
import os.path
from enum import Enum

import matplotlib
import numba
import numpy as np
import pandas as pd
import tqdm
from flask_login import current_user
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist

import TaxonomyForest
import models
from app import db
import Config as C

# Agg engine only supports image generation, which will avoid showing any undesired
# plot windows in the server machine
matplotlib.use('agg')


# Cosine distance
@numba.njit(inline='always', fastmath=True, parallel=True)
def custom_distance(a, b):
    return 1.0 - np.abs(np.dot(a, b) / (np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1.0e-20))


@numba.njit(inline='always', parallel=False, fastmath=True, target_backend='gpu')
def find_bmu(input_vector, weight_vectors) -> int:
    md = 100000000.0
    bmu = -1
    for i in range(0, weight_vectors.shape[0]):
        d = custom_distance(weight_vectors[i, :], input_vector)
        if d <= md:
            md = d
            bmu = i

    return bmu


@numba.njit()
def num_kmember_distance(rec1n, rec2n, num_attribute_ranges):
    return np.abs((rec1n - rec2n) / num_attribute_ranges).sum()


@numba.njit()
def num_kmember_dists_to_centroid(clusterNumArray, CentroidNumArray, num_attribute_ranges):
    n_data_points = clusterNumArray.shape[0]
    distance_to_centroid = np.zeros((n_data_points, 1), dtype=np.float64)

    for i in range(n_data_points):
        nd = np.abs((clusterNumArray[i, :] - CentroidNumArray) / num_attribute_ranges).sum()
        distance_to_centroid[i] = nd

    return distance_to_centroid


class SOM:
    class SOMStatus(Enum):
        NOT_STARTED = 0,
        RUNNING = 1,
        COMPLETED = 2

    net_last_id = 0
    SOMinstances = {}

    def __init__(self, *, datasetID, categoricalFieldMode, sigma, y_class, onlyLoadFields=None):
        net_id: int

        n_data_points: int
        n_highdim_features: int
        n_units: int
        squareLength: int
        num_attr: list
        cat_attr: list
        weight_vectors: np.ndarray
        progress: tqdm.std
        dataFrame: pd.DataFrame
        topology: np.ndarray
        __neighbour_scores: np.ndarray

        # Add SOM to the instance array
        self.net_id = self.net_last_id
        self.process_id = None
        self.net_last_id += 1
        self.SOMinstances[self.net_id] = self
        self.status = SOM.SOMStatus
        self.results = None

        # Internal data
        self.TaxonomyForest = None
        self.progress = None
        self.k_value = 100
        self.a_value = 1.2
        self.b_balue = 0.3
        self.epochs = 0
        self.ensureK = False

        # User parameters
        self.sigma = sigma
        self.categoricalFieldMode = categoricalFieldMode
        self.y_class = y_class.strip()

        # Check if we are going to use a Y-class
        if self.y_class != "":
            self.hasYclass = True
        else:
            self.hasYclass = False

        # Dataset file parameters
        self.datasetID = datasetID
        self.cat_attr = []
        self.num_attr = []
        self.dataFrame = pd.DataFrame()
        self.yDataArray = []

        # Load data from file
        self.loadDatasetFile(onlyLoadFields=onlyLoadFields)

        # Dataset specific parameters obtained from the file
        self.n_data_points = self.dataFrame.values.shape[0]
        self.n_features = self.dataFrame.values.shape[1]

    @staticmethod
    def releaseReferences(SOMID):
        SOM.SOMinstances.pop(SOMID, None)

    def markAsCompleted(self):
        self.status = SOM.SOMStatus.COMPLETED
        if self.results is None:
            # Remove internal references so that it can be destroyed when needed
            self.releaseReferences(self.net_id)

    def loadDatasetFile(self, *, onlyLoadFields=None):
        # Load data
        inFilePath = os.path.join(C.DATASETS_SAVE_PATH, C.DATASET_FILENAME_PREFIX + str(self.datasetID))
        self.dataFrame = pd.read_csv(inFilePath, header=0, encoding='ascii', skipinitialspace=True)

        # Separate Y class if needed
        if self.hasYclass:
            self.yDataArray = self.dataFrame[self.y_class].values
            self.dataFrame = self.dataFrame.drop(self.y_class, axis=1)
        else:
            self.yDataArray = []

        # Drop any unnecesary field if needed
        if onlyLoadFields is not None:
            self.dataFrame = self.dataFrame[onlyLoadFields]

        # Separate categorical and numerical fields
        self.cat_attr = []
        self.num_attr = []
        dtypes = self.dataFrame.dtypes.to_dict()
        for col_name, typ in dtypes.items():
            if typ != 'int64' and typ != 'float64':
                self.cat_attr.append(col_name)
            else:
                self.num_attr.append(col_name)

        # Format values and encode if needed
        self.dataFrame = self.dataFrame.applymap(lambda v: v.strip() if type(v) == str else v)
        if self.categoricalFieldMode == 0 and self.cat_attr != []:
            self.dataFrame[self.cat_attr] = self.dataFrame[self.cat_attr].astype('category')
            self.dataFrame[self.cat_attr] = self.dataFrame[self.cat_attr].apply(lambda x: x.cat.codes)
            self.num_attr = self.num_attr + self.cat_attr
            self.cat_attr = []

        # Drop categorical fields at this point, so that we can normalize values
        self.catDataFrame = self.dataFrame[self.cat_attr]
        self.dataFrame = self.dataFrame.drop(self.cat_attr, axis=1)

        # Order in correct order so distance can later be calculated easier
        if self.categoricalFieldMode == 2:
            self.attrOrder = list(self.cat_attr) + list(self.num_attr)
        else:
            self.attrOrder = list(self.num_attr)
            self.dataFrame = self.dataFrame[self.attrOrder]

        # Normalize values
        self.minNumValues = self.dataFrame.min().values
        self.rangeNumValues = self.dataFrame.max().values - self.minNumValues

        self.dataFrame = (self.dataFrame - self.minNumValues) / self.rangeNumValues

    def neighbourhood_function_gaussian(self, neighbour_index, bmu_index):
        v = norm.pdf(x=neighbour_index, loc=bmu_index, scale=self.sigma)
        v += norm.pdf(x=neighbour_index + self.squareLength, loc=bmu_index, scale=self.sigma)
        v += norm.pdf(x=neighbour_index - self.squareLength, loc=bmu_index, scale=self.sigma)
        return v

    def process_data(self, saveInterval, A=1.2, B=0.3, k_value=50, epochs=None, outFilePath=None, currIterations=0,
                     ensureK=False):
        self.process_id = os.getpid()
        self.status = SOM.SOMStatus.RUNNING
        self.k_value = k_value
        self.a_value = A
        self.b_balue = B
        self.epochs = epochs
        self.ensureK = ensureK

        # Train Network
        self.train(saveInterval, currIterations)

        # Classification
        self.dataFrame = pd.concat([self.catDataFrame, self.dataFrame.copy()], axis=1)

        # No need to continue anymore since we won't save any results
        if outFilePath is None:
            return self.net_id

        self.dataFrame['equivalence_class'] = self.classifyDataPoints()

        if self.hasYclass:
            self.dataFrame[self.y_class] = self.yDataArray

        if self.categoricalFieldMode == 2:
            # Using Taxonomy Trees

            if self.hasYclass:
                Yclass_values = self.dataFrame[self.y_class]
                self.dataFrame = self.dataFrame.drop(self.y_class, axis=1)

            self.TaxonomyForest = TaxonomyForest.TaxonomyForest(dataset_df=self.dataFrame,
                                                                cat_attr=self.cat_attr,
                                                                num_attr=self.num_attr, datasetID=self.datasetID,
                                                                yClass=self.y_class)

            self.dataFrame, centroids = self.TaxonomyForest.adjustment_stage(desired_k=self.k_value)

            for key, centroid in centroids.items():
                centroid[len(self.cat_attr):] = centroid[len(self.cat_attr):] * self.rangeNumValues + self.minNumValues
                centroids[key] = centroid.tolist()

            classificationIDs = self.dataFrame['equivalence_class']
            clusterLabels = classificationIDs.unique()
            clusterSizes = {int(cl_id): len(classificationIDs[classificationIDs == clusterLabels[cl_id]]) for cl_id in
                            range(len(clusterLabels))}

            loss_vector, total_loss = self.TaxonomyForest.information_loss()

            if self.hasYclass:
                self.dataFrame[self.y_class] = Yclass_values

        else:
            # Not using Taxonomy Trees
            if self.hasYclass:
                Yclass_values = self.dataFrame[self.y_class]
                self.dataFrame = self.dataFrame.drop(self.y_class, axis=1)

            clusterLabels = self.dataFrame['equivalence_class'].unique()
            if ensureK:
                self.dataFrame, centroids = self.adjustment_stage()
            else:
                # Generate centroids
                centroids = {
                    int(cl_id):
                        self.dataFrame[self.dataFrame.equivalence_class == clusterLabels[cl_id]].drop(
                            'equivalence_class', axis=1).copy().mean().values
                    for cl_id in range(len(clusterLabels))
                }

            # Scale centroids and convert to list for parsing
            classificationIDs = self.dataFrame['equivalence_class']
            clusterLabels = classificationIDs.unique()
            for key, centroid in centroids.items():
                centroid = centroid * self.rangeNumValues + self.minNumValues
                centroids[key] = centroid.tolist()

            clusterSizes = {int(cl_id): len(classificationIDs[classificationIDs == clusterLabels[cl_id]]) for cl_id in
                            range(len(clusterLabels))}

            loss_vector = {}
            total_loss = 0

            for cl_id in range(len(clusterLabels)):
                data = self.dataFrame[self.dataFrame.equivalence_class == clusterLabels[cl_id]].copy()
                n_points = data.shape[0]

                loss = 0

                # Only calculate loss for numeric values
                for fid in range(len(self.cat_attr), data.shape[1] - 1):
                    v = data.iloc[:, fid].max() - data.iloc[:, fid].min()
                    if np.isnan(v):
                        v = 0.0
                    v = v / self.rangeNumValues[fid - len(self.cat_attr)]
                    loss += v

                loss *= n_points
                loss_vector[int(cl_id)] = loss

                total_loss += loss_vector[cl_id]

            if self.hasYclass:
                self.dataFrame[self.y_class] = Yclass_values

        res = 'None'
        if self.hasYclass:
            corrFrame = self.dataFrame[['equivalence_class', self.y_class]].copy()
            if self.yDataArray.dtype == 'int64' or self.yDataArray.dtype == 'float64':
                res = corrFrame.corr()['equivalence_class'][self.y_class]
            else:
                corrFrame = corrFrame[[self.y_class, 'equivalence_class']]
                corrFrame = corrFrame.astype('category')
                corrFrame = corrFrame.apply(lambda v: v.cat.codes)
                res = corrFrame.corr()['equivalence_class'][self.y_class]

        # Format and rename the equivalence class to cluster
        inFilePath = os.path.join(C.DATASETS_SAVE_PATH, C.DATASET_FILENAME_PREFIX + str(self.datasetID))
        dataset_df_out = pd.read_csv(inFilePath, header=0, encoding='ascii', skipinitialspace=True)

        if self.hasYclass:
            # Add Y class to the order field to avoid losing it.
            self.attrOrder += [self.y_class]

        dataset_df_out = dataset_df_out[self.attrOrder]
        centroidFieldNames = list(dataset_df_out.columns.values)
        if self.hasYclass:
            # Remove Y class from the centroid table
            centroidFieldNames.remove(self.y_class)

        dataset_df_out['cluster'] = self.dataFrame['equivalence_class']
        clusterLabels = dataset_df_out['cluster'].unique()
        for cont in range(len(clusterLabels)):
            dataset_df_out[dataset_df_out.cluster == clusterLabels[cont]].loc[:, 'cluster'] = cont

        dataset_df_out.sort_values(by='cluster', inplace=True)

        # Write results path
        description = f'Hit map'
        self.drawHeatMap(self.hit_map, "HitMap", description, 'winter', self.hit_map.min(), self.hit_map.max())
        self.drawAllSeparateFieldActivations()
        self.results = {'centroids': centroids, 'clusterSizes': clusterSizes, 'nCatFields': len(self.cat_attr),
                        'nTotalDataPoints': self.n_data_points, 'centroidFieldNames': centroidFieldNames,
                        'loss_vector': loss_vector, 'total_loss': total_loss, 'corrRes': res}

        dataset_df_out.to_csv(outFilePath, index=False)

        return self.net_id

    def train(self, saveInterval, startingIteration=0):
        # Based on the desired k-annonymity determine the number of 1D units
        self.squareLength = math.ceil(math.sqrt(self.n_data_points // self.k_value))
        self.n_units = self.squareLength * self.squareLength

        if self.epochs is None or self.epochs == 0:
            self.epochs = int(np.ceil(np.sqrt(self.n_data_points))) // 20

        # Randomly initialized value vectors
        if startingIteration == 0:
            self.weight_vectors = np.random.normal(size=(self.n_units, self.n_features))

        # Prepare matrix to store hits
        self.hit_map = np.zeros((self.n_units,), dtype=np.float32)

        # Calculate positions in square topology
        __neighbour_pos = np.ndarray((self.n_units, 2))
        for cell in range(self.n_units):
            __neighbour_pos[cell, :] = np.array([cell // self.squareLength, cell % self.squareLength])

        # Calculate manhattan distance
        self.__neighbour_scores = 1 - cdist(__neighbour_pos, __neighbour_pos, metric='cityblock')

        # Normalize and apply gaussian scaling according to the selected sigma
        minScor = np.min(self.__neighbour_scores)
        rangScor = np.max(self.__neighbour_scores) - minScor
        self.__neighbour_scores = (self.__neighbour_scores-minScor)/rangScor
        for u in range(self.n_units):
            for v in range(self.n_units):
                self.__neighbour_scores[u][v] *= norm.pdf(x=self.__neighbour_scores[u][v], loc=1, scale=self.sigma)

        # Train the SOM
        totalIterations = self.epochs * self.n_data_points
        itsPerSavePoint = round(totalIterations * saveInterval)
        self.progress = tqdm.trange(startingIteration, totalIterations, initial=startingIteration,
                                    total=totalIterations)
        self.progress.set_description(f'Training SOM (K={self.k_value}, A={self.a_value}, B={self.b_balue})')
        s = startingIteration
        for e in range(self.epochs):
            # Randomly shuffle the frame
            self.dataFrame = self.dataFrame.sample(frac=1)
            epochLimit = (e + 1) * self.n_data_points
            while s < epochLimit:
                # Adjust learning value
                learning_rate = self.a_value ** (-self.b_balue * (s / self.n_data_points))
                input_vector = self.dataFrame.iloc[s % self.n_data_points, :].values

                # Find the best matching unit (lowest distance)
                bmu_idx = find_bmu(input_vector, self.weight_vectors)

                # Go over all possible neighbours and get the score set by the topology
                scores = self.__neighbour_scores[bmu_idx, :]

                # Update hit map
                self.hit_map = self.hit_map + scores

                # Stack values to operate on the complete matrix in a single operation
                scores = np.vstack([scores for _ in range(self.n_features)]).T
                input_vector = np.vstack([input_vector for _ in range(self.n_units)])

                # Update the weight vector according to the SOM formula
                vector_difference = input_vector - self.weight_vectors
                self.weight_vectors = self.weight_vectors + learning_rate * (
                        scores * (1 - (float(s) / (self.n_data_points * self.epochs)))) * vector_difference

                # Update progress
                s += 1
                self.progress.update(1)

                if itsPerSavePoint > 0 and s % itsPerSavePoint == 0:
                    self.save_trainingPoint(s)

    def adjustment_stage(self):
        adjusted_cluster = self.dataFrame['equivalence_class'].copy()

        record_index = np.arange(0, self.dataFrame.shape[0])

        self.dataFrame.set_index(record_index, inplace=True)

        clusterIDs = self.dataFrame['equivalence_class'].unique()
        n_clusters = len(clusterIDs)

        # Generate a dictionary containing the clustered data points, separated according to the SOM results.
        clusters = {
            cl_id: self.dataFrame[self.dataFrame.equivalence_class == clusterIDs[cl_id]].drop('equivalence_class',
                                                                                              axis=1).copy()
            for cl_id in range(n_clusters)}

        # Collect any record in isolated clusters
        self.progress = tqdm.tqdm(clusters.copy().items(), total=n_clusters)
        self.progress.set_description("Collecting spare records from isolated clusters.")
        for cl_id, cluster_data in self.progress:
            if cluster_data.shape[0] >= self.k_value:
                continue

            for i in range(0, cluster_data.shape[0]):
                record = cluster_data.iloc[i, :]
                index = record.__dict__['_name']
                adjusted_cluster[index] = -1
            clusters.pop(cl_id)

        n_clusters = len(clusters)
        # Generate centroids
        centroids = {int(cl_id): cluster.mean().values for cl_id, cluster in clusters.items()}

        # Collect records far away from centroids from big clusters
        self.progress = tqdm.tqdm(clusters.items(), total=n_clusters)
        self.progress.set_description("Collecting far away records from big clusters.")
        for cl_id, cluster_data in self.progress:
            if cluster_data.shape[0] <= self.k_value:
                continue

            # Calculate distance between the centroid and all elements of the cluster
            cluster_data['dist_to_centroid'] = num_kmember_dists_to_centroid(cluster_data[self.num_attr].values,
                                                                             centroids[cl_id].astype(np.float64),
                                                                             self.rangeNumValues)

            # Order so the first element corresponds to the furthest element from the centroid
            cluster_data.sort_values(by='dist_to_centroid', ascending=False, inplace=True)
            cluster_data.drop('dist_to_centroid', axis=1, inplace=True)

            for i in range(0, cluster_data.shape[0] - self.k_value):
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
                if (adjusted_cluster == cluster_id).sum() > self.k_value:
                    continue
                d = num_kmember_distance(record[self.num_attr].astype(np.float64).values,
                                         centroids[cluster_id].astype(np.float64),
                                         self.rangeNumValues)
                if d <= min_d:
                    min_d = d
                    best_cluster = cluster_id

            index = record.__dict__['_name']
            adjusted_cluster[index] = best_cluster

        # Distribute remaining records anywhere
        R = self.dataFrame[adjusted_cluster == -1]
        self.progress = tqdm.trange(R.shape[0])
        self.progress.set_description("Distributing remaining records")
        for i in self.progress:
            record = R.iloc[i, :]
            min_d = 10000000
            best_cluster = -1
            for cluster_id, cluster in clusters.items():
                d = num_kmember_distance(record[self.num_attr].astype(np.float64).values,
                                         centroids[cluster_id].astype(np.float64),
                                         self.rangeNumValues)
                if d <= min_d:
                    min_d = d
                    best_cluster = cluster_id

            index = record.__dict__['_name']
            adjusted_cluster[index] = best_cluster

        self.dataFrame['equivalence_class'] = adjusted_cluster

        return self.dataFrame, centroids

    def save_trainingPoint(self, currIterations):
        date = datetime.datetime.now(datetime.timezone.utc)
        SavePoint = models.TrainedSOM(k_value=self.k_value, epochs=self.epochs, yClass=self.y_class,
                                      a_value=self.a_value, b_value=self.b_balue,
                                      saveDate=date, dataset_id=self.datasetID, sigma=self.sigma,
                                      currIterations=currIterations, categoricalFieldMode=self.categoricalFieldMode
                                      , ensureK=self.ensureK)

        session = db.Session()
        session.add(SavePoint)
        session.commit()
        fileName = C.TRAININGPOINT_FILENAME_PREFIX + str(self.datasetID) + "_" + str(SavePoint.id)
        session.close()

        path = os.path.join(C.DATASETS_TRAINED_POINTS_PATH, fileName)
        pd.DataFrame(self.weight_vectors).to_csv(path, index=False, header=False)

    def load_trainingPointWeights(self, trainingPoint):
        session = db.Session()
        session.query()
        self.datasetID = db.session.query(models.Dataset).filter(
            models.Dataset.id == trainingPoint.dataset_id).first().id
        session.close()
        fileName = C.TRAININGPOINT_FILENAME_PREFIX + str(self.datasetID) + "_" + str(trainingPoint.id)

        path = os.path.join(C.DATASETS_TRAINED_POINTS_PATH, fileName)
        self.weight_vectors = pd.read_csv(path, header=None, encoding='ascii').values

    def classifyDataPoints(self):
        # Find the best matching unit for every input vector and define that as its' cluster id
        out_cluster_ids = pd.Series(np.zeros(self.n_data_points), dtype=int)

        self.progress = tqdm.trange(0, self.n_data_points)
        self.progress.set_description('Classifying data points')
        for i in self.progress:
            input_vector = self.dataFrame.iloc[i, :].loc[self.num_attr].values.astype(float)
            bmu_idx = find_bmu(input_vector, self.weight_vectors)
            out_cluster_ids[i] = bmu_idx

        return out_cluster_ids

    def drawHeatMap(self, R, title, figureText="", colormap="", vmin=0, vmax=1, textColor="red"):
        squareLength = math.ceil(math.sqrt(len(R)))
        D = np.zeros([squareLength, squareLength])

        fig, ax = plt.subplots()
        plt.ioff()

        axis = [str(i).zfill(2) for i in range(0, squareLength)]
        ax.set_xticks(np.arange(squareLength), labels=axis)
        ax.set_yticks(np.arange(squareLength), labels=axis)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        for i in range(0, self.n_units):
            D[i % squareLength][i // squareLength] = R[i]
            if squareLength < 20:
                ax.text(i % squareLength, i // squareLength, round(R[i], 3),
                        ha="center", va="center", color=textColor, fontsize=4)

        if colormap == "":
            colormap = 'summer'

        plot = ax.imshow(D.transpose(), cmap=colormap, vmin=vmin, vmax=vmax)
        fig.colorbar(plot)
        plt.title(title)
        if figureText != "":
            plt.annotate(figureText, (0, 0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top')

        fig.tight_layout()

        fileName = str(title + "." + C.OUTPUT_IMAGE_FORMAT).replace(' ', '_')
        plt.savefig(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id), fileName),
                    bbox_inches='tight', format=C.OUTPUT_IMAGE_FORMAT, dpi=300)

        plt.close()

        return R

    def drawFieldActivation(self, fieldList, dataPointsList=None, nDataPoints=-1, tittle="", description=""):
        R = np.zeros(self.n_units)
        if tittle == "":
            tittle = "Activation for " + str(fieldList)
        if nDataPoints == -1:
            nDataPoints = self.n_data_points

        # Use TENSORDOT INSTEAD OF THIS FOR LOOP
        for fieldName in fieldList:
            fieldIndex = self.num_attr.index(fieldName)
            if dataPointsList is None:
                V = np.dot(self.weight_vectors[:, [fieldIndex]],
                           self.dataFrame.iloc[:, [fieldIndex]].transpose())
            else:
                V = np.dot(self.weight_vectors[:, [fieldIndex]],
                           self.dataFrame.iloc[:, [fieldIndex]][dataPointsList].transpose())

            R = np.add(R, np.sum(V, axis=1) / nDataPoints)

        R = np.divide(R, len(fieldList))
        if description == "":
            description = "Data points: " + str(nDataPoints)

        return self.drawHeatMap(R, tittle, description, 'jet', 0, 1, 'white')

    def drawAllSeparateFieldActivations(self):
        self.progress = tqdm.tqdm(self.num_attr, total=len(self.num_attr))
        self.progress.set_description("Generating graphs")
        for fieldName in self.progress:
            fieldIndex = self.num_attr.index(fieldName)
            V = np.dot(self.weight_vectors[:, [fieldIndex]],
                       self.dataFrame.loc[:, [fieldName]].transpose())

            R = (np.sum(V, axis=1) / self.n_data_points) * self.rangeNumValues[fieldIndex] + self.minNumValues[
                fieldIndex]
            description = f'Min: {self.minNumValues[fieldIndex]} | Max: {self.rangeNumValues[fieldIndex] + self.minNumValues[fieldIndex]} '
            self.drawHeatMap(R, fieldName, description, 'jet', R.min(), R.max(), 'white')

    def drawHypothesisAnalysis(self, causeFields, effectField, HypConf):
        self.dataFrame[self.y_class] = self.yDataArray
        fields = self.cat_attr + self.num_attr
        if effectField not in fields:
            fields += [effectField]

        # Prepare condition strings for each sample
        effectFieldStr = "`" + effectField + "`"
        if HypConf['type'] == 'num':
            numMin = np.min(self.yDataArray)
            numRange = np.max(self.yDataArray) - numMin
            for perc in HypConf['Percentile']:
                HypConf['Num'][int(perc)] = str(max(min(float(HypConf['Num'][int(perc)]), 1), 0) * numRange + numMin)

            # A condition
            conditionA = effectFieldStr + HypConf['Op'][0] + HypConf['Num'][0]
            if HypConf['Op'][1] != 'None':
                conditionA += '&' + effectFieldStr + HypConf['Op'][1] + HypConf['Num'][1]

            # B condition
            conditionB = effectFieldStr + HypConf['Op'][2] + HypConf['Num'][2]
            if HypConf['Op'][3] != 'None':
                conditionB += '&' + effectFieldStr + HypConf['Op'][3] + HypConf['Num'][3]
        else:
            Alabels = []
            Blabels = []
            AllLabels = pd.Series(self.yDataArray).unique()
            for labelIndx in range(len(HypConf['CatLabel'])):
                labelOp = HypConf['CatLabel'][labelIndx]
                if labelOp == '0':  # None
                    continue
                elif labelOp == '1':  # Add to A
                    Alabels.append(AllLabels[labelIndx])
                elif labelOp == '2':  # Add to B
                    Blabels.append(AllLabels[labelIndx])
                elif labelOp == '3':  # Add to Both
                    Alabels.append(AllLabels[labelIndx])
                    Blabels.append(AllLabels[labelIndx])
            conditionA = effectFieldStr + " in " + str(Alabels)
            conditionB = effectFieldStr + " in " + str(Blabels)

        print(conditionA)
        print(conditionB)

        ConditionMatrixA = self.dataFrame.eval(conditionA)
        ConditionMatrixB = self.dataFrame.eval(conditionB)

        NDataPoinstA = np.count_nonzero(ConditionMatrixA)
        NDataPoinstB = np.count_nonzero(ConditionMatrixB)

        descriptionA = "Data points: " + str(NDataPoinstA)
        descriptionB = "Data points: " + str(NDataPoinstB)

        if HypConf['EqualSamples'] is not None:
            # Adjust sample size
            if NDataPoinstA != NDataPoinstB:
                NExcessDataPoints = abs(NDataPoinstA - NDataPoinstB)
                if NDataPoinstA > NDataPoinstB:
                    rndIndex = np.random.choice(ConditionMatrixA[ConditionMatrixA].index.values, NExcessDataPoints,
                                                replace=False)
                    ConditionMatrixA.iloc[rndIndex] = False
                    prevNDataPoints = NDataPoinstA
                    NDataPoinstA = np.count_nonzero(ConditionMatrixA)
                    descriptionA = "Data points: " + str(NDataPoinstA) + " (reduced from " + str(
                        prevNDataPoints) + " matching points)"
                else:
                    rndIndex = np.random.choice(ConditionMatrixB[ConditionMatrixB].index.values, NExcessDataPoints,
                                                replace=False)
                    ConditionMatrixB.iloc[rndIndex] = False
                    prevNDataPoints = NDataPoinstB
                    NDataPoinstB = np.count_nonzero(ConditionMatrixB)
                    descriptionB = "Data points: " + str(NDataPoinstB) + " (reduced from " + str(
                        prevNDataPoints) + " matching points)"

        D = self.drawFieldActivation(causeFields, ConditionMatrixA, NDataPoinstA, "Sample A", descriptionA)
        D = np.subtract(D,
                        self.drawFieldActivation(causeFields, ConditionMatrixB, NDataPoinstB, "Sample B", descriptionB))

        magnitude = np.max(np.abs(D))
        self.drawHeatMap(D, "Difference", ', '.join(causeFields) + " cause: " + effectField, 'RdYlGn', -magnitude,
                         magnitude,
                         "blue")

        return D

    def getProgress(self):
        if self.status == SOM.SOMStatus.NOT_STARTED:
            return {'status': 0}
        elif self.status == SOM.SOMStatus.COMPLETED:
            return {'status': 2}
        else:  # RUNNING
            if self.TaxonomyForest is not None:
                if self.TaxonomyForest.progress is None:
                    return {'status': 0}
                else:
                    progressDict = self.TaxonomyForest.progress.format_dict
                    elapsed = self.progress.format_interval(progressDict["elapsed"])
                    rate = progressDict["rate"]
                    remaining = (
                                        self.TaxonomyForest.progress.total - self.TaxonomyForest.progress.n) / rate if rate and self.TaxonomyForest.progress.total else 0
                    remaining = self.TaxonomyForest.progress.format_interval(remaining)
                    return {'iteration': progressDict['n'], 'totalIterations': progressDict['total'],
                            'rate': rate, 'unit': progressDict['unit'],
                            'elapsed': elapsed, 'remaining': remaining,
                            'desc': progressDict['prefix'], 'status': 1}
            else:
                if self.progress is None:
                    return {'status': 0}
                else:
                    progressDict = self.progress.format_dict
                    elapsed = self.progress.format_interval(progressDict["elapsed"])
                    rate = progressDict["rate"]
                    remaining = (self.progress.total - self.progress.n) / rate if rate and self.progress.total else 0
                    remaining = self.progress.format_interval(remaining)
                    return {'iteration': progressDict['n'], 'totalIterations': progressDict['total'],
                            'rate': rate, 'unit': progressDict['unit'],
                            'elapsed': elapsed, 'remaining': remaining,
                            'desc': progressDict['prefix'], 'status': 1}
