import codecs
import json
import os
import signal

import flask
from flask_login import login_required, current_user
from flask import flash, render_template, request, redirect, url_for, session, current_app, jsonify
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import models as m
from app import db, executor
import Config as C
import SelfOrganizingMap
import Backend

from flask import Blueprint

FrontEndWeb = Blueprint('FrontEndWeb', __name__)


@FrontEndWeb.route('/')
@FrontEndWeb.route('/home/')
@FrontEndWeb.route('/home/<name>')
@login_required
def home(name=None):
    return render_template('home.html', name=name)


@FrontEndWeb.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        dataset = db.session.get(m.Dataset, request.form.get("dataset"))
        if request.form.get("delete") is not None:
            db.session.delete(dataset)
            os.remove(os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName))
        if request.form.get("visibility") is not None:
            dataset.visibility = not dataset.visibility
        if request.form.get("taxonomyTrees") is not None:
            session['selDatasetId'] = dataset.id
            return redirect('/taxonomyTrees')
        if request.form.get("download") is not None:
            session['selDatasetId'] = dataset.id
            return redirect('/download', code=307)
        db.session.commit()

    userDatasets, externalDatasets, ordColumn, ordDirection, ordTable = handleOrderingRequest()
    return render_template('profile.html', user=current_user, userDatasets=userDatasets,
                           externalDatasets=externalDatasets)


@FrontEndWeb.route('/taxonomyTrees', methods=['GET', 'POST'])
@login_required
def taxonomyTrees():
    dataset = db.session.get(m.Dataset, session.get("selDatasetId"))
    if request.method == 'POST':
        treeName = request.form.get('treeName')
        if request.form.get("delete") is not None:
            Backend.DeleteTaxonomyTree(dataset.id, treeName)
            flash("Removed taxonomy tree for field " + treeName + " succesfully", "info")
        if request.form.get("edit") is not None:
            session['taxTreeName'] = treeName
            return redirect(url_for('FrontEndWeb.editTaxonomyTree'))
        if request.form.get("upload") is not None:
            session['selDatasetId'] = dataset.id
            return redirect(url_for('FrontEndWeb.uploadTaxonomyTree'))
        if request.form.get("download") is not None:
            session['selDatasetId'] = dataset.id
            session['taxTreeName'] = treeName
            return redirect('/downloadTaxTree', code=307)
        if request.form.get("generateTrees") is not None:
            session['selDatasetId'] = dataset.id
            session['taxTreeName'] = treeName
            unused, missingTrees = Backend.CheckTaxonomyTrees(dataset.id, True)
            for missingTree in missingTrees:
                flash("A Default Taxonomy Tree for the field " + str(missingTree)
                      + ", was automatically generated", "success")
        # Redirect to display the new state using the GET method
        return redirect(url_for('FrontEndWeb.taxonomyTrees'))

    taxonomyTreeNames, missingTrees = Backend.CheckTaxonomyTrees(dataset.id)

    return render_template('taxonomyTrees.html', name=current_user.name, dataset=dataset,
                           taxonomyTreeNames=taxonomyTreeNames, missingTrees=missingTrees)


@FrontEndWeb.route('/editTaxonomyTree', methods=['GET', 'POST'])
@login_required
def editTaxonomyTree():
    dataset = db.session.get(m.Dataset, session.get("selDatasetId"))
    if request.method == 'POST':
        taxTreeName = request.form.get('taxTreeName')
        if request.form.get('saveTreeText') is not None:
            treeText = request.form.get('treeText')
            treesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(dataset.id))
            fileName = C.TAXONOMY_TREE_FILENAME_PREFIX + taxTreeName + '.' + C.TAXONOMY_TREE_EXTENSION
            filePath = os.path.join(treesDir, fileName)
            with open(filePath, 'w') as f:
                f.write(treeText)
            flash("Changes saved succesfully", 'success')
        return redirect(url_for('FrontEndWeb.editTaxonomyTree'))
    elif request.method == 'GET':
        taxTreeName = session.get('taxTreeName')
        treesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(dataset.id))
        fileName = C.TAXONOMY_TREE_FILENAME_PREFIX + taxTreeName + '.' + C.TAXONOMY_TREE_EXTENSION
        filePath = os.path.join(treesDir, fileName)

        with open(filePath, 'r') as f:
            treeText = f.read()

        return render_template('editTaxonomyTree.html', taxTreeName=taxTreeName, treeText=treeText)


@FrontEndWeb.route('/selectDatasets', methods=['GET', 'POST'])
@login_required
def selectDatasets():
    userDatasets, externalDatasets, ordColumn, ordDirection, ordTable = handleOrderingRequest()
    if request.method == 'GET':
        action = request.args.get('action')
        if action == "SOMAnalysis" or action == "datasetPrivatization":
            nDatasets = 1
        elif action == "compareDatasets":
            nDatasets = 2
        else:
            return "NOT IMPLEMENTED"

        headersToAttributes = {'Dataset': 'datasetName', 'Dimensions': 'dimensions', 'Entries': 'entries', 'Selection': ''}

        return render_template('datasetSelection.html', userDatasets=userDatasets, externalDatasets=externalDatasets,
                               nDatasets=nDatasets, action=action, headersToAttributes=headersToAttributes,
                               ordColumn=ordColumn, ordDirection=ordDirection, ordTable=ordTable)

    elif request.method == 'POST':
        if request.form.get('UploaddDatasets') is not None:
            return redirect(url_for('FrontEndWeb.uploadDatasets'))

        action = request.form.get('action')
        if action == "SOMAnalysis" or action == "datasetPrivatization":
            session['selDatasetId'] = request.form.get('dataset')
        elif action == "compareDatasets":
            session['selDatasetId'] = request.form.getlist('dataset')

        return redirect(url_for("FrontEndWeb." + request.form.get('action')))


@FrontEndWeb.route('/processing', methods=['GET', 'POST'])
@login_required
def processing():
    SOMNetID = session.get('SOMNetID')
    SOMNet = SelfOrganizingMap.SOM.SOMinstances[SOMNetID]
    if request.method == 'POST':
        return redirect(url_for('FrontEndWeb.SOMAnalysis'))

    if SOMNetID is None:
        redirect(url_for('FrontEndWeb.error'))
    dataset = db.session.get(m.Dataset, SOMNet.datasetID)
    return render_template('operationInProcess.html', SOMNetID=SOMNetID, ProgressUpdate=C.PROGRESS_BAR_UPDATE_SPEED,
                           successRedirect=request.args.get('operation'), dataset=dataset)


@FrontEndWeb.route('/SOMAnalysis', methods=['GET', 'POST'])
@login_required
def SOMAnalysis():
    dataset = db.session.get(m.Dataset, session.get('selDatasetId'))
    filePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
    file = Backend.GetDatasetPreview(dataset)
    if request.method == 'GET':
        selectedOperation = request.args.get('selectedOperation')
        NumericalFieldsMask = request.args.get('NumericalFieldsMask', default=None)

        if NumericalFieldsMask is None:
            NumericalFieldsMask = Backend.GetFieldTypes(filePath=filePath)
            NumericalFieldsMaskStr = ""
            for i in range(len(NumericalFieldsMask)):
                NumericalFieldsMaskStr += '1' if NumericalFieldsMask[i] else '0'
            return redirect(url_for('FrontEndWeb.SOMAnalysis', selectedOperation=selectedOperation,
                                    NumericalFieldsMask=NumericalFieldsMaskStr))

        if selectedOperation is None:
            selectedOperation = 'Clustering'

        if selectedOperation == 'Hyperparameter Optimization':
            return render_template('SOMOptimize.html', fileData=file, dataset=dataset,
                                   NumericalFieldsMask=NumericalFieldsMask, selectedOperation=selectedOperation)
        elif selectedOperation == 'Hypothesis Study':
            file = Backend.GetDatasetPreview(dataset)

            CauseFields = request.args.getlist("CauseFields")
            EffectF = request.args.get("EffectF")
            EFind = -1
            EffectLabels = None
            if EffectF is not None:
                EFind = file['fields'].index(EffectF)
                if NumericalFieldsMask[EFind] == '0':
                    EffectLabels = Backend.GetLabelsForField(dataset, EffectF)

            operatorList = ['>', '>=', '==', '!=', '<=', '<']
            return render_template('SOMHypothesis.html', fileData=file, dataset=dataset,
                                   NumericalFieldsMask=NumericalFieldsMask, selectedOperation=selectedOperation,
                                   CauseFields=CauseFields, operatorList=operatorList, EffectF=EffectF, EFind=EFind,
                                   EffectLabels=EffectLabels)
        else:
            trainingPoints = db.session.query(m.TrainedSOM).filter(m.TrainedSOM.dataset_id == dataset.id).all()
            return render_template('SOMCluster.html', fileData=file, dataset=dataset,
                                   NumericalFieldsMask=NumericalFieldsMask, selectedOperation=selectedOperation,
                                   trainingPoints=trainingPoints)

    elif request.method == 'POST':
        selectedOperation = request.form.get('selectedOperation')
        if request.form.get('swapOperation') is not None:
            if selectedOperation == 'Clustering':
                selectedOperation = 'Hyperparameter Optimization'
            elif selectedOperation == 'Hyperparameter Optimization':
                selectedOperation = 'Hypothesis Study'
            elif selectedOperation == 'Hypothesis Study':
                selectedOperation = 'Clustering'
            else:
                selectedOperation = 'Clustering'

            return redirect(url_for('FrontEndWeb.SOMAnalysis', selectedOperation=selectedOperation))

        if request.form.get('categoricalFieldMode') is not None:
            dataset.categoricalFieldMode = (dataset.categoricalFieldMode + 1) % 3
            db.session.commit()
            return redirect(request.url)

        else:
            inFilePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
            categoricalFieldMode = dataset.categoricalFieldMode

            YclassName = request.form.get('YclassName')
            if YclassName == 'None':
                YclassName = ""

            MaxIt = int(request.form.get('MaxIt'))
            sigma = float(request.form.get('sigma'))
            ensureK = True if request.form.get('ensureK') is not None else False

            fileData = Backend.GetDatasetFileData(inFilePath)

            if categoricalFieldMode == 2:  # Taxonomy trees for categorical
                NumericalFieldsMask = request.form.get('NumericalFieldsMask')
                TreesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(dataset.id))
                for i in range(len(NumericalFieldsMask)):
                    if NumericalFieldsMask[i] == '0':
                        FieldName = fileData['fields'][i]
                        TreePath = os.path.join(TreesDir, C.TAXONOMY_TREE_FILENAME_PREFIX + FieldName + '.' +
                                                C.TAXONOMY_TREE_EXTENSION)
                        if not os.path.isfile(TreePath):
                            flash("There is not a valid taxonomy tree for field " + FieldName)
                            return redirect(request.referrer)

            if request.form.get('Hyperparameter Optimization') is not None:
                kMin = int(request.form.get('kMin'))
                kMax = int(request.form.get('kMax'))
                kStep = int(request.form.get('kStep'))
                aMin = float(request.form.get('aMin'))
                aMax = float(request.form.get('aMax'))
                aStep = float(request.form.get('aStep'))
                bMin = float(request.form.get('bMin'))
                bMax = float(request.form.get('bMax'))
                bStep = float(request.form.get('bStep'))

                if categoricalFieldMode != 1:  # Not Omit categorical
                    NumericalFieldsMask = Backend.GetFieldTypes(filePath=filePath)
                    try:
                        NumericalFieldsMask.pop(fileData['fields'].index(YclassName))
                    except ValueError:
                        pass

                    if not any(NumericalFieldsMask):
                        flash("You cannot omit categorical fields, if all the fields in the dataset are categorical or "
                              "the only numerical field is the Y-class.",
                              'danger')
                        return redirect(request.referrer)

                saveInterval = 0

                SOMNetID = Backend.ExhaustiveSomTuning(datasetID=dataset.id, kMin=kMin, kMax=kMax, kStep=kStep,
                                                       aMin=aMin, aMax=aMax, aStep=aStep, bMin=bMin, bMax=bMax,
                                                       bStep=bStep, iterations=MaxIt, sigma=sigma, y_class=YclassName,
                                                       categoricalFieldMode=categoricalFieldMode,
                                                       saveInterval=saveInterval, ensureK=ensureK)
                session['SOMNetID'] = SOMNetID
                return redirect(url_for('FrontEndWeb.processing', operation=url_for('FrontEndWeb.SOMResultT')))

            elif request.form.get('Clustering') is not None:
                if categoricalFieldMode != 1:  # Not Omit categorical
                    NumericalFieldsMask = Backend.GetFieldTypes(filePath=filePath)
                    try:
                        NumericalFieldsMask.pop(fileData['fields'].index(YclassName))
                    except ValueError:
                        pass

                    if not any(NumericalFieldsMask):
                        flash("You cannot omit categorical fields, if all the fields in the dataset are categorical or "
                              "the only numerical field is the Y-class.",
                              'danger')
                        return redirect(request.referrer)

                saveInterval = int(request.form.get('saveInterval')) / 100
                k = int(request.form.get('kValue'))
                if k > dataset.entries:
                    flash("K cannot exceed the number of entries on this dataset ("+str(dataset.entries)+")",
                          'danger')
                    return redirect(request.referrer)
                A = float(request.form.get('aValue'))
                B = float(request.form.get('bValue'))

                outFilePath = os.path.join(C.DATASETS_RESULTS_PATH, current_user.name + '_' + dataset.datasetName)

                SOMNetID = Backend.ProcessClustering(datasetID=dataset.id, k_value=k, A=A, B=B, iterations=MaxIt,
                                                     sigma=sigma, y_class=YclassName,
                                                     categoricalFieldMode=categoricalFieldMode,
                                                     saveInterval=saveInterval, outFilePath=outFilePath,
                                                     ensureK=ensureK)
                session['SOMNetID'] = SOMNetID
                return redirect(url_for('FrontEndWeb.processing', operation=url_for('FrontEndWeb.SOMResultC')))

            elif request.form.get('ClusteringLoad') is not None:
                saveInterval = int(request.form.get('saveInterval')) / 100
                trainingPoint = db.session.get(m.TrainedSOM, request.form.get('trainingPointID'))
                outFilePath = os.path.join(C.DATASETS_RESULTS_PATH, current_user.name + '_' + dataset.datasetName)
                session['SOMNetID'] = Backend.LoadTrainingPoint(trainingPoint=trainingPoint, saveInterval=saveInterval,
                                                                outFilePath=outFilePath)
                return redirect(url_for('FrontEndWeb.processing', operation=url_for('FrontEndWeb.SOMResultC')))

            elif request.form.get('Hypothesis Study') is not None:
                CauseFields = request.form.getlist("CauseFields")

                # =============[ Creater FOLDER FOR EACH USER AND CHECK THAT ]============================
                EffectField = request.form.get('EffectF')
                if EffectField is None:
                    flash("You must select an effect field and set up conditions for it.", "danger")
                    return redirect(request.referrer)
                if len(CauseFields) < 1:
                    flash("You need to add at least 1 cause field", "danger")
                    return redirect(request.referrer)

                if len(CauseFields) != len(set(CauseFields)):
                    flash("You cannot have duplicated cause fields", "danger")
                    return redirect(request.referrer)

                NumericalFieldsMask = request.form.get('NumericalFieldsMask')
                if categoricalFieldMode == 1:  # Encode categorical
                    AllCategorical = True
                    for causeFieldIndx in CauseFields:
                        if NumericalFieldsMask[fileData['fields'].index(causeFieldIndx)] == '1':
                            AllCategorical = False

                    if AllCategorical:
                        flash("You cannot omit categorical fields if all the Cause Fields selected are categorical",
                              'danger')
                        return redirect(url_for('FrontEndWeb.SOMAnalysis', NumericalFieldsMask=NumericalFieldsMask,
                                                selectedOperation=selectedOperation, CauseFields=CauseFields))

                k = int(request.form.get('kValue'))
                if k > dataset.entries:
                    flash("K cannot exceed the number of entries on this dataset ("+str(dataset.entries)+")",
                          'danger')
                    return redirect(request.referrer)
                if NumericalFieldsMask[fileData['fields'].index(EffectField)] == '1':
                    # Numerical effect field
                    HypConf = {'type': 'num',
                               'Op': request.form.getlist('CondOp'),
                               'Num': request.form.getlist('CondNum'),
                               'Percentile': request.form.getlist('CondPerc'),
                               'EqualSamples': request.form.get('EqualSamples')
                               }
                else:
                    # Categorical effect field
                    HypConf = {'type': 'cat',
                               'CatLabel': request.form.getlist('CatLabel'),
                               'EqualSamples': request.form.get('EqualSamples')
                               }
                session['SOMNetID'] = Backend.ProcessHypothesis(datasetID=dataset.id, k_value=k, iterations=MaxIt,
                                                                sigma=sigma, categoricalFieldMode=categoricalFieldMode,
                                                                causeNames=CauseFields, effectName=EffectField,
                                                                HypConf=HypConf, outFilePath=None, ensureK=ensureK)
                return redirect(url_for('FrontEndWeb.processing', operation=url_for('FrontEndWeb.SOMResultH')))
            else:
                CauseFields = request.form.getlist("CauseFields")
                NumericalFieldsMask = request.form.get('NumericalFieldsMask', default=None)
                if request.form.get("AddField") is not None:
                    CauseFields.append('')
                elif request.form.get("RemoveField") is not None:
                    CauseFields.pop()

                if request.form.get("LoadEffectField") is not None:
                    EffectF = YclassName
                    if EffectF in CauseFields:
                        flash("The effect field cannot be the same as the cause fields", "danger")
                        return redirect(request.referrer)
                elif request.form.get("ReloadEffectField") is not None:
                    EffectF = None
                else:
                    EffectF = request.form.get('EffectF')
                return redirect(url_for('FrontEndWeb.SOMAnalysis', NumericalFieldsMask=NumericalFieldsMask,
                                        selectedOperation=selectedOperation, CauseFields=CauseFields, EffectF=EffectF))


@FrontEndWeb.route('/compareDatasets')
@login_required
def compareDatasets():
    datasetIds = session['selDatasetId']
    if len(datasetIds) < 1:
        flash('No datasets selected!', "danger")
        return redirect(url_for('FrontEndWeb.selectDatasets', action='compareDatasets'))
    elif len(datasetIds) > 2:
        flash('You can only select 2 datasets for comparison!', "danger")
        return redirect(url_for('FrontEndWeb.selectDatasets', action='compareDatasets'))
    else:
        datasetA = db.session.get(m.Dataset, datasetIds[0])
        datasetB = db.session.get(m.Dataset, datasetIds[1])
        if datasetA.dimensions != datasetB.dimensions:
            flash('Both datasets must have the same dimensionality!', "danger")
            return redirect(url_for('FrontEndWeb.selectDatasets', action='compareDatasets'))

        if datasetA.datasetName == datasetB.datasetName:
            datasetB.datasetName += '_1'
        files = [Backend.GetDatasetPreview(datasetA, True), Backend.GetDatasetPreview(datasetB, True)]
        results = Backend.CalculateErrors(files[0], files[1])

        return render_template('compareDatasets.html', files=files, results=results)


@FrontEndWeb.route('/operationFinished', methods=['GET', 'POST'])
@login_required
def operationFinished():
    if request.method == 'POST':
        selectedOperation = request.form.get('selectedOperation')
        successUrl = url_for('FrontEndWeb.datasetPrivatization', selectedOperation=selectedOperation)
        if request.form.get("replaceDataset") is not None:
            baseDataset = db.session.get(m.Dataset, session.get('selDatasetId'))
            if baseDataset.user_id != current_user.id:
                flash('You cannot replace a shared dataset. You must do a copy instead.', 'danger')
                return redirect(request.url)

            updatedDataset = db.session.get(m.Dataset, session['tempDatasetId'])

            # Move the updated dataset to replace the old dataset
            oldFilePath = os.path.join(C.DATASETS_TEMPORARY_PATH, updatedDataset.fileName)
            newFilePath = os.path.join(C.DATASETS_SAVE_PATH, baseDataset.fileName)
            os.replace(oldFilePath, newFilePath)

            # Remove the temporary dataset item from the database and update current selection
            baseDataset.dimensions = updatedDataset.dimensions
            baseDataset.entries = updatedDataset.entries
            db.session.delete(updatedDataset)

            db.session.commit()
            session['selDatasetId'] = baseDataset.id
            return redirect(successUrl)
        elif request.form.get("saveDataset") is not None:
            baseDataset = db.session.get(m.Dataset, session.get('selDatasetId'))
            updatedDataset = db.session.get(m.Dataset, session['tempDatasetId'])

            # Ensure we have a valid name and update it
            newName = request.form.get('newDatasetName')
            if newName is None or newName == "":
                flash('You must provide a name for the new dataset!', 'warning')
                return redirect(url_for('FrontEndWeb.operationFinished', selectedOperation=selectedOperation))
            else:
                # Rename, keeping the original extension
                splitName = baseDataset.datasetName.rsplit('.')
                updatedDataset.datasetName = newName + '.' + splitName[1]

            # Move dataset to the permanent storage
            oldFilePath = os.path.join(C.DATASETS_TEMPORARY_PATH, updatedDataset.fileName)
            newFilePath = os.path.join(C.DATASETS_SAVE_PATH, updatedDataset.fileName)
            os.replace(oldFilePath, newFilePath)

            updatedDataset.temporary = False
            db.session.commit()
            session['selDatasetId'] = updatedDataset.id
            return redirect(successUrl)
    if request.method == 'GET':
        selectedOperation = request.args.get('selectedOperation')
        dataset = db.session.get(m.Dataset, session.get('selDatasetId'))
        canReplace = current_user.id == dataset.user_id
        return render_template('operationFinished.html', selectedOperation=selectedOperation, canReplace=canReplace)


@FrontEndWeb.route('/datasetPrivatization', methods=['GET', 'POST'])
@login_required
def datasetPrivatization():
    dataset = db.session.get(m.Dataset, session.get('selDatasetId'))

    if request.method == 'POST':
        selectedOperation = request.form.get('selectedOperation')
        if request.form.get('swapOperation') is not None:
            if selectedOperation == 'Swap':
                selectedOperation = 'Noise'
            elif selectedOperation == 'Noise':
                selectedOperation = 'Strip'
            elif selectedOperation == 'Strip':
                selectedOperation = 'Generalize'
            else:
                selectedOperation = 'Swap'
            return redirect(url_for('FrontEndWeb.datasetPrivatization', selectedOperation=selectedOperation))
        if selectedOperation == 'Swap':
            swapRate = request.form.get("SwapRate")
            swapFields = request.form.getlist("SwapFields")
            groupFields = True if request.form.get("GroupFields") is not None else False
            if request.form.get("AddField") is not None:
                swapFields.append('')
            elif request.form.get("RemoveField") is not None:
                swapFields.pop()
            elif request.form.get("Operate") is not None:
                filePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
                creationError, newId = Backend.CreateNewDataset(sourcePath=filePath, datasetName=dataset.datasetName,
                                                                visibility=dataset.visibility, userId=current_user.id,
                                                                temporaryCopy=True)
                if creationError is None:
                    session['tempDatasetId'] = newId
                    dataset = db.session.get(m.Dataset, newId)
                    Backend.SwapData(dataset=dataset, swapRate=float(swapRate) / 100.0, swapFields=swapFields,
                                     groupFields=groupFields)
                    flash('Operation completed successfully!', 'success')
                else:
                    flash(creationError, 'danger')
                return redirect(url_for('FrontEndWeb.operationFinished', selectedOperation=selectedOperation))
            return redirect(url_for('FrontEndWeb.datasetPrivatization', SwapRate=swapRate, SwapFields=swapFields,
                                    GroupFields=groupFields, selectedOperation=selectedOperation))
        elif selectedOperation == 'Noise':
            if request.form.get('swapNoise') is not None:
                selectedNoise = request.form.get('selectedNoise')
                if selectedNoise == 'Kim':
                    selectedNoise = 'Uniform'
                else:
                    selectedNoise = 'Kim'
                return redirect(url_for('FrontEndWeb.datasetPrivatization', selectedOperation=selectedOperation,
                                        selectedNoise=selectedNoise))
            if request.form.get("Operate") is not None:
                selectedNoise = request.form.get("selectedNoise")
                if selectedNoise is None:
                    flash("Error with the selected noise type", 'danger')
                    return redirect(url_for('FrontEndWeb.error'))

                filePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
                creationError, newId = Backend.CreateNewDataset(sourcePath=filePath, datasetName=dataset.datasetName,
                                                                visibility=dataset.visibility, userId=current_user.id,
                                                                temporaryCopy=True)
                if creationError is not None:
                    flash(creationError, 'danger')
                    return redirect(url_for('FrontEndWeb.error'))

                if selectedNoise == 'Uniform':
                    affectedRate = request.form.get("affectedRate")
                    minNoiseRate = request.form.get("minNoiseRate")
                    maxNoiseRate = request.form.get("maxNoiseRate")

                    session['tempDatasetId'] = newId
                    dataset = db.session.get(m.Dataset, newId)
                    Backend.AddUniformNoise(dataset=dataset, affectedRate=float(affectedRate) / 100,
                                            minNoiseRate=float(minNoiseRate) / 100,
                                            maxNoiseRate=float(maxNoiseRate) / 100)
                    flash('Operation completed successfully!', 'success')
                elif selectedNoise == 'Kim':
                    affectedRate = request.form.get("affectedRate")
                    noiseMagnitude = request.form.get("noiseMagnitude")

                    session['tempDatasetId'] = newId
                    dataset = db.session.get(m.Dataset, newId)
                    Backend.AddKimNoise(dataset=dataset, affectedRate=float(affectedRate) / 100,
                                        noiseMagnitude=float(noiseMagnitude) / 100)
                    flash('Operation completed successfully!', 'success')
                return redirect(url_for('FrontEndWeb.operationFinished', selectedOperation=selectedOperation))
        elif selectedOperation == 'Strip':
            selFields = request.form.getlist("SelFields")
            if request.form.get("AddField") is not None:
                selFields.append('')
            elif request.form.get("RemoveField") is not None:
                selFields.pop()
            elif request.form.get("Operate") is not None:
                filePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
                creationError, newId = Backend.CreateNewDataset(sourcePath=filePath, datasetName=dataset.datasetName,
                                                                visibility=dataset.visibility, userId=current_user.id,
                                                                temporaryCopy=True)
                if creationError is None:
                    session['tempDatasetId'] = newId
                    dataset = db.session.get(m.Dataset, newId)
                    newShape = Backend.StripFields(dataset=dataset, selFields=selFields)
                    dataset.entries = newShape[0]
                    dataset.dimensions = newShape[1]
                    db.session.commit()
                    flash('Operation completed successfully!', 'success')
                else:
                    flash(creationError, 'danger')
                return redirect(url_for('FrontEndWeb.operationFinished', selectedOperation=selectedOperation))
            return redirect(url_for('FrontEndWeb.datasetPrivatization', SelFields=selFields,
                                    selectedOperation=selectedOperation))
        elif selectedOperation == 'Generalize':
            divisions = request.form.get("divisions")
            if divisions is None:
                divisionList = request.form.getlist("DivisionList")
            else:
                divisionList = [divisions]

            selFields = request.form.getlist("SelFields")
            if request.form.get("AddField") is not None:
                selFields.append('')
                divisionList.append('2')
            elif request.form.get("RemoveField") is not None:
                selFields.pop()
                divisionList.pop()
            elif request.form.get("Operate") is not None:
                if len(selFields) != len(set(selFields)):
                    flash("You cannot select duplicated fields", "danger")
                    return redirect(request.referrer)

                filePath = os.path.join(C.DATASETS_SAVE_PATH, dataset.fileName)
                creationError, newId = Backend.CreateNewDataset(sourcePath=filePath, datasetName=dataset.datasetName,
                                                                visibility=dataset.visibility, userId=current_user.id,
                                                                temporaryCopy=True)
                if creationError is None:
                    session['tempDatasetId'] = newId
                    dataset = db.session.get(m.Dataset, newId)
                    Backend.GeneralizeFields(dataset=dataset, selFields=selFields, divisionList=divisionList)
                    flash('Operation completed successfully!', 'success')
                else:
                    flash(creationError, 'danger')
                return redirect(url_for('FrontEndWeb.operationFinished', selectedOperation=selectedOperation))
            return redirect(url_for('FrontEndWeb.datasetPrivatization', SelFields=selFields, DivisionList=divisionList,
                                    selectedOperation=selectedOperation))

    elif request.method == 'GET':
        file = Backend.GetDatasetPreview(dataset)

        selectedOperation = request.args.get('selectedOperation')
        if selectedOperation is None:
            selectedOperation = 'Swap'

        if selectedOperation == 'Swap':
            swapRate = request.args.get("SwapRate")
            if swapRate is None:
                swapRate = 30
            swapFields = request.args.getlist("SwapFields")
            groupFields = True if request.args.get("GroupFields") is not None else False
            return render_template('PrivatizationSwap.html', fileData=file, datasetName=dataset.datasetName,
                                   selectedOperation=selectedOperation, swapRate=swapRate, swapFields=swapFields,
                                   groupFields=groupFields)
        elif selectedOperation == 'Noise':
            selectedNoise = request.args.get("selectedNoise")
            if selectedNoise is None:
                selectedNoise = 'Kim'
            return render_template('PrivatizationNoise.html', fileData=file, datasetName=dataset.datasetName,
                                   selectedOperation=selectedOperation, selectedNoise=selectedNoise)
        elif selectedOperation == 'Strip':
            selFields = request.args.getlist("SelFields")
            return render_template('PrivatizationStrip.html', fileData=file, datasetName=dataset.datasetName,
                                   selectedOperation=selectedOperation, selFields=selFields)
        elif selectedOperation == 'Generalize':
            selFields = request.args.getlist("SelFields")
            divisions = request.args.get("divisions")
            if divisions is None:
                divisionList = request.args.getlist("DivisionList")
                if len(divisionList) == 0:
                    divisionList = [2]
            else:
                divisionList = [divisions]

            return render_template('PrivatizationGeneralize.html', fileData=file, datasetName=dataset.datasetName,
                                   selectedOperation=selectedOperation, divisionList=divisionList, selFields=selFields)
        else:
            return "OPERATION NOT IMPLEMENTED"


@FrontEndWeb.route('/uploadDatasets', methods=['GET', 'POST'])
@login_required
def uploadDatasets():
    if request.method == 'GET':
        return render_template('uploadDatasets.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', "error")
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No file selected!', "danger")
            return redirect(request.url)
        if file and allowed_datasetFile(file.filename):
            # Get file data and save the file in the temporary directory
            datasetName = secure_filename(file.filename)
            visibility = False if request.form.get('public', 'off') == 'off' else True

            filePath = os.path.join(current_app.config['UPLOAD_FOLDER'], datasetName)
            file.save(filePath)

            # Create new file in the system and check for errors
            error, unusedId = Backend.CreateNewDataset(sourcePath=filePath, datasetName=datasetName,
                                                       visibility=visibility, userId=current_user.id)
        else:
            error = "The selected file extension is not supported"

        if error is None:
            # Render the page
            flash('File uploaded successfully!', "success")
            return redirect(request.url)
        else:
            # Display the error
            flash(error, "danger")
            return redirect(request.url)


@FrontEndWeb.route('/uploadTaxonomyTree', methods=['POST'])
@login_required
def uploadTaxonomyTree():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', "error")
            return redirect(request.referrer)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No file selected!', "danger")
            return redirect(request.referrer)
        if file and allowed_taxonomyTreeFile(file.filename):
            # Get file data and save the file in the temporary directory
            dataset = db.session.get(m.Dataset, session['selDatasetId'])
            taxonomyTreeName = session['taxTreeName']
            treesDir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(dataset.id))
            fileName = C.TAXONOMY_TREE_FILENAME_PREFIX + taxonomyTreeName + '.' + C.TAXONOMY_TREE_EXTENSION

            filePath = os.path.join(treesDir, fileName)
            file.save(filePath)
            flash('File uploaded successfully!', "success")
            return redirect(request.referrer)
        else:
            flash("The selected file extension is not supported", "danger")
            return redirect(request.referrer)


@FrontEndWeb.route('/SOMResultC', methods=['GET', 'POST'])
def SOMResultC():
    if request.method == 'POST':
        if request.form.get('downloadAnalysis') is not None:
            return redirect('/downloadAnalysis', code=307)
        elif request.form.get('downloadClustered') is not None:
            return redirect('/downloadClustered', code=307)
        else:
            flash('Something went wrong with the request', 'danger')
            return redirect(url_for('FrontEndWeb.error'))

    elif request.method == 'GET':
        SOMNetID = session.get('SOMNetID')
        results = Backend.GetResults(int(SOMNetID))

        dataset = db.session.get(m.Dataset, int(session.get('selDatasetId')))
        file_path = os.path.join(C.DATASETS_TEMPORARY_PATH, current_user.name + '_' + dataset.datasetName)
        if results is not None:
            json.dump(results, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'),
                      sort_keys=True,
                      indent=4)
        else:
            obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
            results = json.loads(obj_text)

        if results is None:
            flash('We could not locate the results of this operation', 'danger')
            return redirect(url_for('FrontEndWeb.error'))
        else:
            path = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id))
            imageNames = next(os.walk(path), (None, None, []))[2]  # [] if no file
            return render_template('SOMClusterResult.html', dataset=dataset,
                                   centroids=results['centroids'], clusterSizes=results['clusterSizes'],
                                   nTotalDataPoints=results['nTotalDataPoints'], corrRes=results['corrRes'],
                                   centroidFieldNames=results['centroidFieldNames'],
                                   loss_vector=results['loss_vector'], total_loss=results['total_loss'],
                                   nCatFields=results['nCatFields'], imageNames=imageNames, nimages=len(imageNames))


@FrontEndWeb.route('/SOMResultH', methods=['GET'])
def SOMResultH():
    SOMNetID = session.get('SOMNetID')
    dataset = db.session.get(m.Dataset, int(session.get('selDatasetId')))

    path = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id))
    imageNames = next(os.walk(path), (None, None, []))[2]  # [] if no file
    results = Backend.GetResults(int(SOMNetID))
    return render_template('ShowGraphs.html', imageNames=imageNames, nimages=len(imageNames))


@FrontEndWeb.route('/SOMResultT', methods=['GET'])
def SOMResultT():
    SOMNetID = session.get('SOMNetID')
    dataset = db.session.get(m.Dataset, int(session.get('selDatasetId')))

    path = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id))
    imageNames = next(os.walk(path), (None, None, []))[2]  # [] if no file
    results = Backend.GetResults(int(SOMNetID))
    return render_template('ShowGraphs.html', imageNames=imageNames, nimages=len(imageNames))


@FrontEndWeb.route('/status/<SOMNetID>', methods=['GET'])
def getStatus(SOMNetID):
    responseDict = Backend.GetProgress(int(SOMNetID))
    return json.dumps(responseDict)


@FrontEndWeb.route('/image')
def image():
    filename = request.args.get('filename')
    if filename is None:
        return None
    else:
        path = os.path.join(C.FULL_IMAGES_SAVE_PATH, str(current_user.id))
        return flask.send_from_directory(path, filename)


@FrontEndWeb.route('/download', methods=['POST'])
def download():
    dataset = db.session.get(m.Dataset, session['selDatasetId'])
    return flask.send_from_directory(C.DATASETS_SAVE_PATH, dataset.fileName, as_attachment=True,
                                     download_name=dataset.datasetName)


@FrontEndWeb.route('/downloadTaxTree', methods=['POST'])
def downloadTaxTree():
    dataset = db.session.get(m.Dataset, session['selDatasetId'])
    taxonomyTreeName = session['taxTreeName']
    serving_dir = os.path.join(C.TAXONOMY_TREE_PATH, C.TAXONOMY_TREE_DIRECTORY_PREFIX + str(dataset.id))
    fileName = C.TAXONOMY_TREE_FILENAME_PREFIX + taxonomyTreeName + '.' + C.TAXONOMY_TREE_EXTENSION
    return flask.send_from_directory(serving_dir, fileName, as_attachment=True, download_name=fileName)


@FrontEndWeb.route('/downloadAnalysis', methods=['POST'])
def downloadAnalysis():
    dataset = db.session.get(m.Dataset, int(session.get('selDatasetId')))
    fileName = current_user.name + '_' + dataset.datasetName
    return flask.send_from_directory(C.DATASETS_TEMPORARY_PATH, fileName, as_attachment=True,
                                     download_name=fileName + '_Result.json')


@FrontEndWeb.route('/downloadClustered', methods=['POST'])
def downloadClustered():
    dataset = db.session.get(m.Dataset, int(session.get('selDatasetId')))
    splitName = dataset.datasetName.rsplit('.')
    fileName = current_user.name + '_' + dataset.datasetName
    return flask.send_from_directory(C.DATASETS_RESULTS_PATH, fileName, as_attachment=True,
                                     download_name=splitName[0] + C.DATASET_CLUSTERED_SUFFIX + '.' + splitName[1])


def handleOrderingRequest():
    ordParams = request.args.get('order')
    if ordParams is not None:
        ordParams, ordTable = os.path.splitext(ordParams)
        ordColumn, ordDirection = os.path.splitext(ordParams)
        ordDirection = int(ordDirection[1:])
        ordTable = int(ordTable[1:])

        ordDirection = (ordDirection + 1) % 3
        if ordDirection == 0:
            ordColumn = None

        userDatasets, extDatasets = Backend.GetUserDatasets(current_user.id, ordColumn, ordDirection < 2)
    else:
        ordColumn = ""
        ordDirection = 0
        ordTable = 0
        userDatasets, extDatasets = Backend.GetUserDatasets(current_user.id)

    return userDatasets, extDatasets, ordColumn, ordDirection, ordTable


@FrontEndWeb.errorhandler(404)
def pageNotFound(error):
    return "page not found"


@FrontEndWeb.errorhandler(Exception)
def handle_error(e):
    return redirect(url_for('FrontEndWeb.error'))


@FrontEndWeb.route('/error', methods=['GET'])
def error():
    return render_template('error.html')


def errorToJson(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code


@FrontEndWeb.route('/favicon.ico')
def favicon():
    basePath, unused = os.path.split(FrontEndWeb.root_path)
    return flask.send_from_directory(os.path.join(basePath, 'FlaskApp', 'favicon_io'), 'favicon.ico')


@FrontEndWeb.route('/apple-touch-icon.ico')
def apple_icon():
    basePath, unused = os.path.split(FrontEndWeb.root_path)
    return flask.send_from_directory(os.path.join(basePath, 'FlaskApp', 'favicon_io'), 'apple-touch-icon.png')


@FrontEndWeb.route('/icon')
def icon():
    basePath, unused = os.path.split(FrontEndWeb.root_path)
    return flask.send_from_directory(os.path.join(basePath, 'FlaskApp', 'favicon_io'), 'favicon-32x32.png')


@FrontEndWeb.route('/manifest')
def manifest():
    basePath, unused = os.path.split(FrontEndWeb.root_path)
    return redirect(os.path.join(basePath, 'FlaskApp', 'favicon_io', 'site.webmanifest'))


def allowed_datasetFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in C.ALLOWED_DATASET_EXTENSIONS


def allowed_taxonomyTreeFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == C.TAXONOMY_TREE_EXTENSION
