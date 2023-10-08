import csv
import enum
import functools
import os
import shutil

import concurrent.futures

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LightSource
from sklearn.decomposition import PCA
from flask_login import current_user

import models
from app import db, executor
import Config as C
import SelfOrganizingMap


class Operation(enum.Enum):
    CLUSTERING = 0,
    TUNING = 1,
    HYPOTHESIS = 2


def CreateNewDataset(*, sourcePath, datasetName, visibility, userId, temporaryCopy=False):
    datasetId = -1

    # Get file data
    data = GetDatasetFileData(sourcePath)
    error = CheckDatasetValidity(sourcePath)

    if error is None:
        # Create dataset item for the database
        dataset = models.Dataset(datasetName=datasetName,
                                 fileName=sourcePath,
                                 visibility=visibility,
                                 user_id=userId,
                                 categoricalFieldMode=0,
                                 dimensions=data['nfields'],
                                 entries=data['nentries'],
                                 temporary=temporaryCopy)

        # Add file to the database and generate unique ID
        db.session.add(dataset)
        db.session.commit()

        # Move file to the storage directory and rename based on the ID
        dataset.fileName = C.DATASET_FILENAME_PREFIX + str(dataset.id)

        if temporaryCopy:
            newFilePath = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
            shutil.copy(sourcePath, newFilePath)
        else:
            newFilePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
            os.replace(sourcePath, newFilePath)

        db.session.commit()
        datasetId = dataset.id

    return error, datasetId


def GetFieldTypes(*, matrix=None, filePath=None, inverse=False):
    if filePath is not None:
        with open(filePath, newline='') as csvFile:
            reader = csv.reader(csvFile, delimiter=',', quotechar='|', skipinitialspace=True)

            data = list(reader)
            matrix = np.array(data)

            # Eliminate header, which contains field names
            matrix = np.delete(matrix, 0, 0)

    M = matrix
    nFields = M.shape[1]
    NumericalFieldsMask = [not inverse] * nFields
    M = np.delete(M, 0, 0)
    for i in range(0, nFields):
        try:
            M[:, i].astype(float)
        except ValueError:
            NumericalFieldsMask[i] = inverse

    return NumericalFieldsMask


def CheckTaxonomyTrees(datasetID, generateDefaultTrees=False):
    treesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(datasetID))
    os.makedirs(treesDir, 0o777, True)

    datasetPath = os.path.join(C.DATASETS_SAVE_PATH, C.DATASET_FILENAME_PREFIX + str(datasetID))
    dataFrame = pd.read_csv(datasetPath, header=0, encoding='ascii', skipinitialspace=True)

    # Separate categorical and numerical fields
    taxonomyTreeNames = []
    dtypes = dataFrame.dtypes.to_dict()
    for col_name, typ in dtypes.items():
        if typ != 'int64' and typ != 'float64':
            taxonomyTreeNames.append(col_name)

    missingTrees = []

    for tree in taxonomyTreeNames:
        path = os.path.join(treesDir, C.TAXONOMY_TREE_FILENAME_PREFIX + tree + '.' + C.TAXONOMY_TREE_EXTENSION)
        if os.path.isfile(path) is False:
            missingTrees.append(tree)
            if generateDefaultTrees:
                labels = dataFrame.loc[:, tree].unique()
                with open(path, 'w') as f:
                    for label in labels:
                        f.write(label + ";*" + '\n')

    return taxonomyTreeNames, missingTrees


def DeleteTaxonomyTree(datasetID, name):
    treesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(datasetID))
    filePath = os.path.join(treesDir, C.TAXONOMY_TREE_FILENAME_PREFIX + name + '.' + C.TAXONOMY_TREE_EXTENSION)
    os.remove(filePath)


def CheckDatasetValidity(filePath):
    error = None
    with open(filePath, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='|', skipinitialspace=True)
        header = next(reader)
        nHeaderValues = len(header)
        if nHeaderValues != len(set(header)):
            return "Header contains duplicated column names"

        for field in header:
            if field.find('.') != -1:
                return "Header cannot have column names containing a dot"

        for row in reader:
            nDataValues = len(row)
            if nDataValues == 0:
                return "File contains blank lines, which must be removed"
            if nDataValues != nHeaderValues:
                return f'Row {reader.line_num} of the file has {nDataValues} values, but the dataset dimensionality is {nHeaderValues}. Use dots to indicate decimal numbers and commas to separate each value in the row. '

    return error


def GetUserDatasets(user_id, orderAttrName=None, ascend=True, separateExternal=False):
    if orderAttrName is None:
        orderAttr = models.Dataset.id
    else:
        orderAttr = getattr(models.Dataset, orderAttrName)

    if ascend:
        # Ascending order
        userDatasets = db.session.query(models.Dataset).filter(models.Dataset.temporary.is_(False)).filter(
            models.Dataset.user_id == user_id).order_by(orderAttr).all()

        externalDatasets = db.session.query(models.Dataset).filter(models.Dataset.temporary.is_(False)).filter(
            models.Dataset.user_id != user_id, models.Dataset.visibility == 1).order_by(orderAttr).all()

    else:
        # Descending order
        userDatasets = db.session.query(models.Dataset).filter(models.Dataset.temporary.is_(False)).filter(
            models.Dataset.user_id == user_id).order_by(orderAttr.desc()).all()

        externalDatasets = db.session.query(models.Dataset).filter(models.Dataset.temporary.is_(False)).filter(
            models.Dataset.user_id != user_id, models.Dataset.visibility == 1).order_by(orderAttr.desc()).all()

    return userDatasets, externalDatasets


def GetDatasetPreview(dataset, addStatistics=False):
    fileData = {'name': dataset.datasetName}
    with open(os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName), newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='|', skipinitialspace=True)
        row = next(reader)
        fileData['fields'] = row
        fileData['nfields'] = len(row)

        fileData['entries'] = []
        nPreviewEntries = 0
        while row is not None and nPreviewEntries <= C.WEB_MAX_PREVIEW_ENTRIES:
            row = next(reader)
            fileData['entries'].append(row)
            nPreviewEntries = nPreviewEntries + 1

        fileData['nPreviewEntries'] = min(C.WEB_MAX_PREVIEW_ENTRIES, reader.line_num)

    if addStatistics:
        df = pd.read_csv(os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName), header=0, encoding='ascii',
                         skipinitialspace=True)
        fileData['nentries'] = df.shape[0]
        fileData['NumericalFieldMask'] = GetFieldTypes(matrix=df.values)
        Mean, Variance = CalculateStatistics(df, fileData['NumericalFieldMask'])

        fileData['MEAN'] = list(Mean)
        fileData['VARIANCE'] = list(Variance)
        fileData['PCA'] = GetPCA(df, fileData['NumericalFieldMask'], fileData['name'])

    for i in range(len(fileData['entries'])):
        for j in range(len(fileData['entries'][i])):
            try:
                fileData['entries'][i][j] = round(float(fileData['entries'][i][j]), 3)
                if fileData['entries'][i][j].is_integer():
                    fileData['entries'][i][j] = int(fileData['entries'][i][j])
            except ValueError:
                continue

    return fileData


def GetLabelsForField(dataset, fieldName):
    path = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
    df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)
    if df[fieldName].dtypes != 'object':
        return None
    else:
        return df[fieldName].unique()


def GetDatasetFileData(path):
    fileData = {'name': os.path.basename(path)}
    data = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)
    fileData['fields'] = list(data.columns)
    fileData['nfields'] = len(fileData['fields'])
    fileData['nentries'] = data.shape[0]
    fileData['nPreviewEntries'] = min(C.WEB_MAX_PREVIEW_ENTRIES, fileData['nentries'])

    return fileData


def CalculateErrors(FileDataA, FileDataB):
    CRA = np.corrcoef(FileDataA['PCA']['rawComponents'])[0]
    CRB = np.corrcoef(FileDataB['PCA']['rawComponents'])[0]

    Cells = CRA.shape[0]
    MSE = 0.0
    MAE = 0.0
    MV = 0.0
    diff = 0.0
    for i in range(0, Cells):
        diff = CRA[i] - CRB[i]
        MSE += diff * diff / Cells
        MAE += abs(diff) * abs(diff) / Cells

    for i in range(0, Cells):
        MV += diff / CRA[i] / Cells

    return {'MSE': MSE, 'MAE': MAE, 'MV': MV}


def CalculateStatistics(M, NumericalFieldMask):
    Mean = np.ndarray(M.shape[1], dtype=object)
    Mean[NumericalFieldMask] = np.mean(M.loc[:, NumericalFieldMask].astype(float), axis=0).round(decimals=2)
    Variance = np.var(M.loc[:, NumericalFieldMask].astype(float), axis=0).round(decimals=2)

    CategoricalFieldMask = [not val for val in NumericalFieldMask]
    Frequency = np.ndarray(len(np.nonzero(CategoricalFieldMask)[0]), dtype=object)
    cont = 0
    for i in range(len(CategoricalFieldMask)):
        if CategoricalFieldMask[i]:
            unique, counts = np.unique(M.iloc[:, i], return_counts=True)
            Frequency[cont] = unique[counts.argmax()] + ' (' + str(round(np.max(counts) / M.shape[0] * 100, 2)) + '%)'
            cont += 1

    Mean[CategoricalFieldMask] = Frequency

    return Mean, Variance


def GetPCA(df, NumericalFieldsMask, datasetName, n_pcs=0):
    # Eliminate non-numeric columns
    df = df[df.columns[NumericalFieldsMask]].astype(float)

    # number of components
    if n_pcs < 1:
        n_pcs = df.shape[1]

    model = PCA(n_components=n_pcs).fit(df)
    M = model.transform(df)

    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    most_important_names = [df.columns[most_important[i]] for i in range(n_pcs)]

    most_important = {
        'PC{}'.format(i): (most_important_names[i], round(abs(model.components_[i][most_important[i]] * 100), 3)) for i
        in range(n_pcs)}

    PlotPCA(M[:, 0:2], np.transpose(model.components_[0:2, :]), datasetName)

    return {'expVarRat': np.around(model.explained_variance_ratio_ * 100, 3), 'mostImportant': most_important,
            'components': np.around(np.abs(model.components_) * 100, decimals=3),
            'rawComponents': model.components_,
            'featureNames': model.feature_names_in_}


def PlotPCA(score, coeff, datasetName, labels=None):
    plt.figure()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0] // 200
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='red')
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    datasetName = datasetName.split('.')[0] + '.' + C.OUTPUT_IMAGE_FORMAT
    plt.savefig(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id), datasetName),
                bbox_inches='tight', format=C.OUTPUT_IMAGE_FORMAT, dpi=300)


def RemovePreviousImages():
    # New dataset operation should remove previous images
    path = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id))
    imageDir = os.scandir(path)
    for entry in imageDir:
        if entry.is_file():
            file_path = entry.path
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    imageDir.close()


def ExhaustiveSomTuning(*, datasetID, iterations, sigma, y_class, categoricalFieldMode,
                        kMin, kMax, kStep, aMin, aMax, aStep, bMin, bMax, bStep, saveInterval, ensureK):
    RemovePreviousImages()
    SOMNet = SelfOrganizingMap.SOM(datasetID=datasetID, categoricalFieldMode=categoricalFieldMode,
                                   sigma=sigma, y_class=y_class)

    if saveInterval > 0:
        saveInterval = 1  # Save only on last iteration

    kRange = int((kMax - kMin) // kStep) + 1
    aRange = int((aMax - aMin) // aStep) + 1
    bRange = int((bMax - bMin) // bStep) + 1
    SOMNet.results = {'ResM': np.zeros((kRange, aRange, bRange))}

    executor.submit(ExhaustiveSomTuningIter,
                    **{'SOMNet': SOMNet, 'iterations': iterations, 'saveInterval': saveInterval,
                       'kMin': kMin, 'kMax': kMax, 'kStep': kStep, 'aMin': aMin, 'aMax': aMax, 'aStep': aStep,
                       'bMin': bMin, 'bMax': bMax, 'bStep': bStep, 'ensureK': ensureK})

    return SOMNet.net_id


def ExhaustiveSomTuningIter(SOMNet, iterations, saveInterval, kMin, kMax, kStep, aMin, aMax, aStep, bMin, bMax, bStep,
                            ensureK):
    kRange = int((kMax - kMin) // kStep) + 1
    aRange = int((aMax - aMin) // aStep) + 1
    bRange = int((bMax - bMin) // bStep) + 1
    for kInd in range(kRange):
        for aInd in range(aRange):
            for bInd in range(bRange):
                k_value = kMin + kStep * kInd
                A = aMin + aStep * aInd
                B = bMin + bStep * bInd

                SOMNet.loadDatasetFile()
                SOMNet.process_data(saveInterval=saveInterval, k_value=k_value, A=A, B=B, epochs=iterations,
                                    ensureK=ensureK)

                SOMNet.dataFrame[SOMNet.y_class] = SOMNet.yDataArray

                if SOMNet.yDataArray.dtype == 'int64' or SOMNet.yDataArray.dtype == 'float64':
                    res = SOMNet.dataFrame.corr()['equivalence_class'][SOMNet.y_class]
                else:
                    SOMNet.dataFrame = SOMNet.dataFrame[[SOMNet.y_class, 'equivalence_class']]
                    SOMNet.dataFrame = SOMNet.dataFrame.astype('category')
                    SOMNet.dataFrame = SOMNet.dataFrame.apply(lambda v: v.cat.codes)
                    res = SOMNet.dataFrame.corr()['equivalence_class'][SOMNet.y_class]

                SOMNet.results['ResM'][kInd][aInd][bInd] = res

    M = SOMNet.results['ResM']
    D = M.shape

    # Draw the images
    for i in range(D[0]):
        # Load and format data
        z = M[i]
        nrows, ncols = z.shape
        x = np.linspace(aMin, aMax, ncols)
        y = np.linspace(bMin, bMax, nrows)
        x, y = np.meshgrid(x, y)

        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        fileName = f'K_{i * kStep + kMin}'
        plt.title(f'K = {i * kStep + kMin}')
        plt.xlabel('A')
        plt.ylabel('B')

        ls = LightSource(270, 45)
        rgb = ls.shade(z, cmap=cm.coolwarm, vert_exag=0.1, blend_mode='soft')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)

        plt.savefig(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id), fileName), bbox_inches='tight',
                    format=C.OUTPUT_IMAGE_FORMAT,
                    dpi=300)

    SOMNet.markAsCompleted()


def FinishedExahustiveSomTuning(future, kStep):
    # Fetch net object
    netID = future.result()
    SOMNet = SelfOrganizingMap.SOM.SOMinstances[netID]

    # Calculate last correlation
    SOMNet.dataFrame[SOMNet.y_class] = SOMNet.dataFrame[SOMNet.y_class]
    corr = SOMNet.dataFrame.corr()['SOM'][SOMNet.y_class]

    M = SOMNet.results['CorrM']
    D = M.shape

    SOMNet.results['CorrM'][D[0]][D[1]][D[2] + 1] = corr

    # Draw the images
    for i in range(D[0]):
        # Load and format data
        z = M[i]
        nrows, ncols = z.shape
        x = np.linspace(0, 1, ncols)
        y = np.linspace(0, 1, nrows)
        x, y = np.meshgrid(x, y)

        region = np.s_[5:50, 5:50]
        x, y, z = x[region], y[region], z[region]

        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        fileName = f'K_{i * kStep}.' + C.OUTPUT_IMAGE_FORMAT
        plt.title(f'K = {i * kStep}')
        plt.xlabel('A')
        plt.ylabel('B')

        ls = LightSource(270, 45)
        rgb = ls.shade(z, cmap=cm.coolwarm, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)

        plt.savefig(os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id), fileName), bbox_inches='tight',
                    format=C.OUTPUT_IMAGE_FORMAT,
                    dpi=300)

    SOMNet.markAsCompleted()


def ProcessClustering(*, datasetID, k_value, A, B, iterations, sigma, y_class, categoricalFieldMode, saveInterval,
                      outFilePath, ensureK):
    RemovePreviousImages()
    SOMNet = SelfOrganizingMap.SOM(datasetID=datasetID, categoricalFieldMode=categoricalFieldMode,
                                   sigma=sigma, y_class=y_class)

    future = executor.submit(SOMNet.process_data, **{'saveInterval': saveInterval, 'k_value': k_value, 'A': A, 'B': B,
                                                     'outFilePath': outFilePath, 'epochs': iterations,
                                                     'ensureK': ensureK})
    future.add_done_callback(FinishedClusteringTrain)

    return SOMNet.net_id


def FinishedClusteringTrain(future):
    netID = future.result()
    Net = SelfOrganizingMap.SOM.SOMinstances[netID]
    Net.markAsCompleted()


def ProcessHypothesis(*, datasetID, k_value, iterations, sigma, categoricalFieldMode, causeNames,
                      effectName, HypConf, outFilePath, ensureK):
    saveInterval = 0  # Disabled for these operations
    RemovePreviousImages()
    SOMNet = SelfOrganizingMap.SOM(datasetID=datasetID, categoricalFieldMode=categoricalFieldMode,
                                   sigma=sigma, y_class=effectName, onlyLoadFields=causeNames)

    future = executor.submit(SOMNet.process_data, **{'saveInterval': saveInterval, 'k_value': k_value,
                                                     'outFilePath': outFilePath, 'epochs': iterations,
                                                     'ensureK': ensureK})

    future.add_done_callback(functools.partial(FinishedHypothesisTrain, causeNames, effectName, HypConf))

    return SOMNet.net_id


def FinishedHypothesisTrain(causeNames, effectName, HypConf, future):
    netID = future.result()
    Net = SelfOrganizingMap.SOM.SOMinstances[netID]
    RemovePreviousImages()
    description = f'Hit map'
    Net.drawHeatMap(Net.hit_map, "HitMap", description, 'winter', Net.hit_map.min(), Net.hit_map.max())
    M = Net.drawHypothesisAnalysis(causeNames, effectName, HypConf)
    Net.results = {'Matrix': M}
    Net.markAsCompleted()


def LoadTrainingPoint(*, trainingPoint, saveInterval, outFilePath):
    SOMNet = SelfOrganizingMap.SOM(datasetID=trainingPoint.dataset_id,
                                   categoricalFieldMode=trainingPoint.categoricalFieldMode,
                                   sigma=trainingPoint.sigma, y_class=trainingPoint.yClass)
    SOMNet.load_trainingPointWeights(trainingPoint)
    future = executor.submit(SOMNet.process_data, **{'saveInterval': saveInterval, 'k_value': trainingPoint.k_value,
                                                     'A': trainingPoint.a_value, 'B': trainingPoint.b_value,
                                                     'outFilePath': outFilePath, 'epochs': trainingPoint.epochs,
                                                     'currIterations': trainingPoint.currIterations,
                                                     'ensureK': trainingPoint.ensureK})
    future.add_done_callback(FinishedClusteringTrain)

    return SOMNet.net_id


def GetProgress(SOM_ID):
    Net = SelfOrganizingMap.SOM.SOMinstances.get(SOM_ID)
    if Net is None:
        # SOM net does not exist, probably destroyed already
        return {'status': -1}
    else:
        return Net.getProgress()


def GetResults(SOM_ID, *, disposeNet=True):
    Net = SelfOrganizingMap.SOM.SOMinstances.get(SOM_ID)
    if Net is None:
        return None
    else:
        results = Net.results
        if disposeNet:
            SelfOrganizingMap.SOM.releaseReferences(SOM_ID)
        return results


def SwapData(*, dataset, swapRate, swapFields, groupFields):
    path = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
    nSwaps = round(dataset.entries * swapRate)
    if nSwaps % 2 > 0:
        nSwaps = nSwaps - 1

    dataset_df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)
    fieldNames = list(dataset_df.columns)
    swapFieldsIndx = []
    for swapField in swapFields:
        swapFieldsIndx.append(fieldNames.index(swapField))

    if groupFields:
        rndIndex = np.random.choice(dataset.entries, nSwaps, replace=False)
        rndIndex = np.reshape(rndIndex, [-1, 2])
        dataset_df.iloc[rndIndex[:, 0], swapFieldsIndx] = dataset_df.iloc[rndIndex[:, 1], swapFieldsIndx]

    dataset_df.to_csv(path, index=False)


def AddUniformNoise(*, dataset, affectedRate, minNoiseRate, maxNoiseRate):
    path = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
    dataset_df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)
    NumericalFieldsMask = GetFieldTypes(matrix=dataset_df.values)

    numeric_ranges = dataset_df.loc[:, dataset_df.columns[NumericalFieldsMask]].max(axis=0) - dataset_df.loc[:,
                                                                                              dataset_df.columns[
                                                                                                  NumericalFieldsMask]].min(
        axis=0)

    nAffectedRows = round(dataset.entries * affectedRate)
    rndIndex = np.random.choice(dataset.entries, nAffectedRows, replace=False)

    minVal = minNoiseRate * numeric_ranges[0]
    rangeVal = (maxNoiseRate * numeric_ranges[0] - minVal)
    noise = np.random.rand(nAffectedRows, 1) * rangeVal + minVal
    for i in range(1, len(numeric_ranges)):
        minVal = minNoiseRate * numeric_ranges[i]
        rangeVal = (maxNoiseRate * numeric_ranges[i] - minVal)
        noise = np.concatenate((noise, np.random.rand(nAffectedRows, 1) * rangeVal + minVal), axis=1)

    dataset_df.loc[rndIndex, dataset_df.columns[NumericalFieldsMask]] += (noise - noise / 2)
    dataset_df.to_csv(path, index=False)


def AddKimNoise(*, dataset, affectedRate, noiseMagnitude):
    path = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
    dataset_df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)
    NumericalFieldsMask = GetFieldTypes(matrix=dataset_df.values)

    numeric_vars = dataset_df.loc[:, dataset_df.columns[NumericalFieldsMask]].std()

    nAffectedRows = round(dataset.entries * affectedRate)
    rndIndex = np.random.choice(dataset.entries, nAffectedRows, replace=False)

    noise = np.random.normal(0, noiseMagnitude * numeric_vars, (nAffectedRows, len(numeric_vars)))

    dataset_df.loc[rndIndex, dataset_df.columns[NumericalFieldsMask]] += noise
    dataset_df.to_csv(path, index=False)


def StripFields(*, dataset, selFields):
    path = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
    dataset_df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)

    dataset_df.drop(selFields, axis=1, inplace=True)

    dataset_df.to_csv(path, index=False)

    return dataset_df.shape


def GeneralizeFields(*, dataset, selFields, divisionList):
    path = os.path.join(C.DATASETS_TEMPORARY_PATH, dataset.fileName)
    dataset_df = pd.read_csv(path, header=0, encoding='ascii', skipinitialspace=True)

    # If there are no selected fields, assume all of the fields will be affected by the same division amount
    if len(selFields) < 1:
        selFields = list(dataset_df.columns.values)
        divisionList = divisionList * len(selFields)

    # Separate categorical and numerical fields of the selected fields
    cat_attr = []
    num_attr = []
    dtypes = dataset_df[selFields].dtypes.to_dict()
    for col_name, typ in dtypes.items():
        if typ != 'int64' and typ != 'float64':
            cat_attr.append(col_name)
        else:
            num_attr.append(col_name)

    for column in num_attr:
        divisions = int(divisionList[selFields.index(column)])
        colMax = dataset_df[column].max()
        colMin = dataset_df[column].min()
        divisionSpace = np.linspace(colMin, colMax, num=divisions + 1)
        divisionLimits = list(zip(divisionSpace, divisionSpace[1:]))
        divisionLabels = np.ndarray(shape=(len(divisionLimits),), dtype=object)
        for i in range(len(divisionLabels)):
            divisionLabels[i] = "[" + str(divisionLimits[i][0]) + "-" + str(divisionLimits[i][1]) + (
                ")" if i < len(divisionLabels) - 1 else "]")
        labelIndx = (dataset_df[column] - colMin) // ((colMax - colMin) / divisions)
        labelIndx[labelIndx == divisions] -= 1
        dataset_df[column] = divisionLabels[labelIndx.values.astype(int)]

    for column in cat_attr:
        divisions = int(divisionList[selFields.index(column)])
        categories = dataset_df[column].unique()
        categories = list(np.random.choice(categories, len(categories), replace=False))  # Random order
        groupLength = max(len(categories) // divisions, 2)
        excess = len(categories) % groupLength
        divisionGroups = list(zip(*(iter(categories),) * groupLength))
        divisionGroups[-1] += tuple(categories[-excess:])
        for i in range(len(divisionGroups)):
            divisionName = '|'.join(divisionGroups[i])
            for label in divisionGroups[i]:
                dataset_df[column] = dataset_df[column].where(dataset_df[column] == label, divisionName)

    dataset_df.to_csv(path, index=False)
