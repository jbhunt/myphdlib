import numpy as np
import pathlib as pl
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from myphdlib.general.toolkit import resample
from myphdlib.extensions.matplotlib import (
    SaccadeDirectionLabelingGUI,
    SaccadeEpochLabelingGUI,
 )

# TODO
# [X] Allow for appending new training data instead of overwritting
# [X] Implement scikit-learn's multioutput model for predicting saccade epochs
# [X] Record classification results agnostic of any parameter (i.e., don't split on direction of saccade)
# [X] Save saccade  epochs as one dataset (right now I'm saving the onset/offset separately)

def _extendTrainingDataset(
    session,
    head,
    xNew,
    yNew,
    zNew=None,
    overwrite=False
    ):
    """
    """

    #
    name = pl.Path(head).name
    xPath = f'prediction/saccades/{name}/X'
    yPath = f'prediction/saccades/{name}/y'
    zPath = f'prediction/saccades/{name}/z'

    #
    if session.hasDataset(head):
        if overwrite:
            for path in (xPath, yPath, zPath):
                session.remove(path)
            xOld = np.empty([0, xNew.shape[1]]).astype(xNew.dtype)
            yOld = np.empty([0, yNew.shape[1]]).astype(yNew.dtype)
        else:
            xOld = session.load(xPath)
            yOld = session.load(yPath)
    else:
        xOld = np.empty([0, xNew.shape[1]]).astype(xNew.dtype)
        yOld = np.empty([0, yNew.shape[1]]).astype(yNew.dtype)

    #
    X = np.vstack([xOld, xNew])
    y = np.vstack([yOld, yNew])

    # TODO: Remove duplicate samples

    #
    session.save(xPath, X)
    session.save(yPath, y)

    #
    if zNew is not None:
        if session.hasDataset(zPath) and overwrite:
            zOld = session.load(zPath)
        else:
            zOld = np.empty([0, zNew.shape[1]]).astype(yNew.dtype)
        z = np.vstack([zOld, zNew])
        session.save(zPath, z)

    return

class PredictionProcessingMixin(object):
    """
    """


    def _labelSaccadeWaveforms(self, nSamples=1, overwrite=False):
        """
        Manually score the direction of a subset of putative saccades (nasal/temporal/noise)
        """

        #
        saccadeWaveformsPutative = list()
        peakVelocities = list()
        for eye in ('left', 'right'):
            saccadeWaveformsPutative_ = self.load(f'saccades/putative/{eye}/waveforms')
            if saccadeWaveformsPutative_ is None:
                continue

            for sample in saccadeWaveformsPutative_:
                if np.isnan(sample).sum() != 0:
                    continue
                saccadeWaveformsPutative.append(sample)
                peakVelocities.append(np.diff(sample).max())

        #
        saccadeWaveformsPutative = np.array(saccadeWaveformsPutative)
        sampleIndices = np.random.choice(
            np.arange(saccadeWaveformsPutative.shape[0]),
            size=nSamples,
        )

        #
        gui = SaccadeDirectionLabelingGUI()
        gui.inputSamples(saccadeWaveformsPutative[sampleIndices, :])
        while gui.isRunning():
            continue
        

        # Append new training data
        X, y = gui.trainingData
        _extendTrainingDataset(
            self,
            head='prediction/saccades/direction',
            xNew=X,
            yNew=y,
            overwrite=overwrite
        )

        return

    def _labelSaccadeEpochs(self, nSamples=1, overwrite=False, gain=1.5):
        """
        Manually label saccade onset and offset
        """

        # Collect waveforms and labels for all saccades
        saccadeWaveforms = list()
        saccadeLabels = list()
        for eye in ('left', 'right'):
            saccadeWaveforms_ = self.load(f'saccades/predicted/{eye}/waveforms')
            if saccadeWaveforms_ is None:
                continue
            saccadeLabels_ = self.load(f'saccades/predicted/{eye}/labels')
            for sampleIndex in np.arange(saccadeWaveforms_.shape[0]):
                saccadeWaveforms.append(saccadeWaveforms_[sampleIndex])
                saccadeLabels.append(saccadeLabels_[sampleIndex])
        saccadeWaveforms = np.array(saccadeWaveforms)
        saccadeLabels = np.array(saccadeLabels).reshape(-1, 1)

        #
        sampleIndices = np.random.choice(
            np.arange(saccadeWaveforms.shape[0]),
            size=nSamples
        )
        gui = SaccadeEpochLabelingGUI()
        gui.inputSamples(saccadeWaveforms[sampleIndices, :], saccadeLabels[sampleIndices], gain)
        while gui.isRunning():
            continue

        # Append the new training data
        X, y, z = gui.trainingData
        _extendTrainingDataset(
            self,
            head='prediction/saccades/epochs',
            xNew=X,
            yNew=y,
            zNew=z,
            overwrite=overwrite
        )

        return

def _trainSaccadeDirectionClassifier(
    sessions,
    nFeatures=30,
    classifier='mlp',
    ):
    """
    """

    #
    xTrain = list()
    xTest = list()
    yTrain = list()

    #
    for session in sessions:

        #
        for eye in ('left', 'right'):
            saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{eye}/waveforms')
            if saccadeWaveformsUnlabeled is None:
                continue
            for x in saccadeWaveformsUnlabeled:
                if np.isnan(x).any():
                    continue
                t, xp = resample(np.diff(x), nFeatures)
                xTest.append(xp)

        #
        saccadeWaveformsLabeled = session.load(f'prediction/saccades/direction/X')
        if saccadeWaveformsLabeled is None:
            continue
        labels = session.load(f'prediction/saccades/direction/y')
        nSamples = labels.shape[0]
        session.log(f'Collected training data for predicting saccade epochs ({nSamples} samples)')
        iterable = zip(saccadeWaveformsLabeled, labels)
        for x, label in iterable:
            if np.isnan(x).any():
                continue
            t, xp = resample(np.diff(x), nFeatures)
            xTrain.append(xp)
            yTrain.append(label.item())

    #
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)

    # Fit
    nSamples = xTrain.shape[0]
    print(f'INFO: Training saccade direction classifier ({nSamples} samples)')
    if classifier == 'mlp':
        grid = {
            'hidden_layer_sizes': [
                (10 , 10 , 10 ),
                (100, 100, 100),
                (200, 200, 200),
                (300, 300, 300)
            ],
            'max_iter': [
                1000000,
            ]
        }
        net = MLPClassifier()
        search = GridSearchCV(net, grid)
        search.fit(xTrain, yTrain.ravel())
        clf = search.best_estimator_
    
    #
    elif classifier == 'lda':
        lda = LinearDiscriminantAnalysis()
        lda.fit(xTrain, yTrain.ravel())
        clf = lda

    return clf, xTrain, yTrain

def _trainSaccadeEpochRegressor(
    sessions,
    nFeatures=30,
    saccadeDirection=1,
    ):
    """
    """

    #
    xTrain = list()
    xTest = list()
    yTrain = list()

    #
    for session in sessions:

        #
        for eye in ('left', 'right'):
            saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{eye}/waveforms')
            if saccadeWaveformsUnlabeled is None:
                continue
            for x in saccadeWaveformsUnlabeled:
                if np.isnan(x).any():
                    continue
                t, xp = resample(np.diff(x), nFeatures)
                xTest.append(xp)

        #
        saccadeWaveformsLabeled = session.load(f'prediction/saccades/epochs/X')
        if saccadeWaveformsLabeled is None:
            continue
        saccadeEpochLabels = session.load(f'prediction/saccades/epochs/y')
        saccadeDirections = session.load(f'prediction/saccades/epochs/z')
        sampleIndices = np.where(saccadeDirections == saccadeDirection)[0]
        nSamples = sampleIndices.size
        session.log(f'Collected training data for predicting saccade epochs ({nSamples} samples)')
        iterable = zip(saccadeWaveformsLabeled[sampleIndices, :], saccadeEpochLabels[sampleIndices, :])
        for x, label in iterable:
            if np.isnan(x).any():
                continue
            t, xp = resample(np.diff(x), nFeatures)
            xTrain.append(xp)
            yTrain.append(label)

    #
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)

    #
    nSamples = xTrain.shape[0]
    print(f'INFO: Training saccade epoch regressor ({nSamples} samples)')
    grid = {
        'estimator__hidden_layer_sizes': [
            (10 , 10 , 10 ),
            (100, 100, 100),
            (200, 200, 200),
            (300, 300, 300),
        ],
        'estimator__max_iter': [
            1000000,
        ]
    }
    reg = MultiOutputRegressor(MLPRegressor())
    search = GridSearchCV(reg, grid)
    search.fit(xTrain, yTrain)
    reg = search.best_estimator_

    return reg, xTrain, yTrain


def predictSaccadeDirection(
    sessionsToAnalyze,
    sessionsForTraining,
    nFeatures=30,
    classifier='mlp',
    ):
    """
    """

    #
    clf, xTrain, yTrain = _trainSaccadeDirectionClassifier(
        sessionsForTraining,
        nFeatures,
        classifier,
    )

    #
    for session in sessionsToAnalyze:
        for eye in ('left', 'right'):
            xTest, saccadeWaveforms, frameIndices = list(), list(), list()
            saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{eye}/waveforms')
            saccadeIndicesUnlabeled = session.load(f'saccades/putative/{eye}/indices')
            for saccadeIndex, saccadeWaveform in enumerate(saccadeWaveformsUnlabeled):
                x = np.diff(saccadeWaveform)
                if np.isnan(x).sum() != 0:
                    continue
                t, xp = resample(x, nFeatures)
                xTest.append(xp)
                frameIndices.append(saccadeIndicesUnlabeled[saccadeIndex])
                saccadeWaveforms.append(saccadeWaveform)
            xTest = np.array(xTest)
            saccadeWaveforms = np.array(saccadeWaveforms)
            frameIndices = np.array(frameIndices)
            yPredicted = clf.predict(xTest)

            #
            nSaccades = 0
            for saccadeCode, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):
                nSaccades += np.sum(yPredicted == saccadeCode)
            session.log(f'{nSaccades} saccades predicted from the {eye} eye')

            #
            saccadeMask = np.logical_or(
                yPredicted == -1,
                yPredicted == +1
            )

            #
            session.save(f'saccades/predicted/{eye}/indices', frameIndices[saccadeMask])
            session.save(f'saccades/predicted/{eye}/waveforms', saccadeWaveforms[saccadeMask])
            session.save(f'saccades/predicted/{eye}/labels', yPredicted[saccadeMask])

    return

# TODO: Log some kind of metric for the saccade epochs prediction (maybe mean epoch size)
def predictSaccadeEpochs(
    sessionsToAnalyze,
    sessionsForTraining,
    nFeatures=30,
    ):
    """
    """

    # Train the regressors
    regressors = {
        'nasal': None,
        'temporal': None,
    }
    for saccadeDirection in (-1, 1):
        reg, xTrain, yTrain = _trainSaccadeEpochRegressor(
            sessionsForTraining,
            nFeatures,
            saccadeDirection
        )
        if saccadeDirection == -1:
            regressors['nasal'] = reg
        else:
            regressors['temporal'] = reg

    # Loop through each session
    for session in sessionsToAnalyze:

        #
        saccadeEpochs = {
            'left': None,
            'right': None
        }

        # Loop through each eye
        for eye in ('left', 'right'):

            # Load datasets
            saccadeLabels = session.load(f'saccades/predicted/{eye}/labels')
            saccadeWaveforms = session.load(f'saccades/predicted/{eye}/waveforms')
            frameIndices = session.load(f'saccades/predicted/{eye}/indices').reshape(-1, 1)
            nSaccades = saccadeLabels.shape[0]
            saccadeEpochs[eye] = np.full([nSaccades, 2], np.nan)

            #
            for saccadeDirection in (-1, 1):

                #
                k = 'nasal' if saccadeDirection == -1 else 'temporal'
                reg = regressors[k]

                #
                saccadeIndices = np.where(saccadeLabels == saccadeDirection)[0]
                xTest = np.full([saccadeIndices.size, nFeatures], np.nan)

                #
                for sampleIndex, saccadeIndex in enumerate(saccadeIndices):
                    x = np.diff(saccadeWaveforms[sampleIndex, :])
                    if np.isnan(x).sum() != 0:
                        continue
                    t, xp = resample(x, nFeatures)
                    xTest[sampleIndex, :] = xp

                #
                yPredicted = reg.predict(xTest) + frameIndices[saccadeIndices, :]
                saccadeEpochs[eye][saccadeIndices, :] = yPredicted

        #
        for eye in ('left', 'right'):
            session.save(f'saccades/predicted/{eye}/epochs', saccadeEpochs[eye])

    return
