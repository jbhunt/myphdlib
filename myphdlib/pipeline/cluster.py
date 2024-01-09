import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import pandas as pd

class ClusterProcessingMixin():
    """
    """

    def _extractResponseWaveforms(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        normalizingWindow=(-0.2, 0.5),
        minimumBaselineActivity=2,
        minimumResponseAmplitude=2,
        binsize=0.02,
        overwrite=False,
        nBins=101,
        sd=0.01,
        ):
        """
        """

        self.log(f'Extracting event-related response waveforms')

        #
        nUnits = self.population.count()

        #
        datasetKeys = (
            ('probe', 'preferred'),
            ('probe', 'nonpreferred'),
            ('saccade', 'preferred'),
            ('saccade', 'nonpreferred')
        )

        # Check if data is alread processed 
        flags = np.full(4, False)
        for i, (eventType, eventDirection) in enumerate(datasetKeys):
            datasetPath = f'peths/{eventType}/{eventDirection}'
            if self.hasDataset(datasetPath):
                flags[i] = True
        if flags.all() and overwrite == False:
           return

        # Skip sessions without the drifting grating stimulus
        if self.probeTimestamps is None:
            for eventType, eventDirection in datasetKeys:
                datasetPath = f'peths/{eventType}/{eventDirection}'
                self.save(datasetPath, np.full([nUnits, nBins], np.nan))
            return

        # Event type, event direction, event timestamps
        iterable = (
            ('probe', self.probeTimestamps, self.gratingMotionDuringProbes),
            ('saccade', self.saccadeTimestamps[:, 0], self.saccadeLabels)
        )

        #
        for eventType, eventTimestamps, eventDirections in iterable:

            self.log(f'Extracting preferred and non-preferred responses to {eventType}s')

            #
            peths = {
                'preferred': np.full([nUnits, nBins], np.nan),
                'nonpreferred': np.full([nUnits, nBins], np.nan)
            }

            #
            nUnits = self.population.count()
            for iUnit, unit in enumerate(self.population):

                self.log(f'Working on unit {iUnit + 1} out of {nUnits}')

                #
                factors = list()
                responses = list()
                baselines = list()
                prominences = list()

                #
                for eventDirection in (-1, +1):

                    # Estimate baseline activity
                    t, M = psth2(
                        eventTimestamps[eventDirections == eventDirection],
                        unit.timestamps,
                        window=baselineWindow,
                        binsize=None
                    )
                    mu = M.flatten().mean() / np.diff(baselineWindow).item()

                    # Filter out units with subthreshold baselines
                    if mu < minimumBaselineActivity:
                        factors.append(np.nan)
                        responses.append(np.full(nBins, np.nan))
                        baselines.append(np.nan)
                        prominences.append(np.nan)
                        continue
                    else:
                        baselines.append(mu)

                    # Measure the evoked response
                    try:
                        t, fr = unit.peth(
                            eventTimestamps[eventDirections == eventDirection],
                            responseWindow=responseWindow,
                            binsize=binsize,
                            kde=True,
                            sd=sd,
                        )
                    except:
                        import pdb; pdb.set_trace()

                    # Filter out units with subthreshold baselines
                    if np.max(np.abs(fr - mu)) < minimumResponseAmplitude:
                        factors.append(np.nan)
                        responses.append(np.full(nBins, np.nan))
                        baselines.append(np.nan)
                        prominences.append(np.nan)
                        continue

                    # Save the baseline subtracted response
                    normalizingWindowMask = np.logical_and(
                        t >= normalizingWindow[0],
                        t <= normalizingWindow[1]
                    )
                    responses.append(fr - mu)
                    factor = round(np.max(np.abs(fr - mu)[normalizingWindowMask]), 2)
                    factors.append(factor)

                    # Measure the prominence of the response
                    prominence = round(np.max(np.abs(fr - mu)) - np.mean(fr - mu), 2)
                    prominences.append(prominence)

                # Skip unit if either response fails the minimum baseline activity threshold
                if np.isnan(factors).sum() != 0:
                    peths['preferred'][iUnit, :] = np.full(nBins, np.nan)
                    peths['nonpreferred'][iUnit, :] = np.full(nBins, np.nan)
                    continue

                # Normalize and determine preference
                iP = np.argmax(prominences)
                iN = 0 if iP == 1 else 1
                factor = factors[iP]
                rP = np.around(responses[iP] / factor, 2)
                rN = np.around(responses[iN] / factor, 2)

                # Store the peths
                peths['preferred'][iUnit, :] = rP
                peths['nonpreferred'][iUnit, :] = rN

            #
            for k in peths.keys():
                self.save(f'peths/{eventType}/{k}', peths[k])

        return

    def _runClusterModule(self):
        """
        """

        self._extractResponseWaveforms()

        return

def clusterUnitsByVisualResponseShape(
    sessions,
    k=None,
    kmin=1,
    kmax=15,
    model='gmm',
    ):
    """
    """

    #
    for session in sessions:
        session.population.unfilter()

    # Collect samples
    samples = list()
    inclusionMasksBySession = list()
    for session in sessions:
        peths = session.load(f'peths/probe/preferred')
        inclusionMask = np.invert(np.isnan(peths).all(1))
        inclusionMasksBySession.append(inclusionMask)
        for unit in session.population:
            sample = peths[unit.index, :]
            if np.isnan(sample).all():
                continue
            else:
                samples.append(sample)
    X = np.array(samples)

    #
    if k is None:
        krange = np.arange(kmin, kmax + 1, 1)
    else:
        krange = None

    # Hierarchical clustering
    if model == 'agg':
        if k is None:
            grid = {
                'n_clusters': krange,
            }
            search = GridSearchCV(
                AgglomerativeClustering(),
                param_grid=grid,
                scoring=lambda estimator, X: -1 * silhouette_score(X, estimator.labels_)
            )
            search.fit(X)
            model = search.best_estimator_
            scores = -1 * search.cv_results_['mean_test_score']
        else:
            model = AgglomerativeClustering(n_clusters=k).fit(X)
            scores = None
        labels = model.labels_

    # Gaussian mixture model
    elif model == 'gmm':
        if k is None:
            grid = {
                'n_components': krange,
            }
            search = GridSearchCV(
                GaussianMixture(max_iter=10000, covariance_type='diag'),
                param_grid=grid,
                scoring=lambda estimator, X: -1 * estimator.bic(X)
            )
            search.fit(X)
            model = search.best_estimator_
            scores = -1 * search.cv_results_['mean_test_score']
        else:
            model = GaussianMixture(n_components=k).fit(X)
            scores = None
        labels = model.predict(X)

    #
    xDecomposed = PCA(n_components=2).fit_transform(X)

    #
    nClusters = np.unique(labels).size
    print(f'INFO: Visual responses optimally fit to {nClusters} clusters')

    # NOTE: This needs to be done for estimators that don't implement a 'predict' method
    nUnitsIncludedBySession = np.array([
        m.sum() for m in inclusionMasksBySession
    ])
    splitIndices = np.cumsum(nUnitsIncludedBySession)[:-1]
    iterable = zip(
        np.split(labels, splitIndices),
        np.split(xDecomposed, splitIndices, axis=0)
    )
    for sessionIndex, (clusterLabels, unitCoords) in enumerate(iterable):

        #
        session = sessions[sessionIndex]
        inclusionMask = inclusionMasksBySession[sessionIndex]

        # Save predicted cluster labels
        clusterLabels_= np.full(session.population.count(), np.nan)
        clusterLabels_[inclusionMask] = clusterLabels
        session.save(f'clustering/probe/labels', clusterLabels_)

        # Save subspace coordinates
        unitCoords_ = np.full([session.population.count(), 2], np.nan)
        unitCoords_[inclusionMask, :] = unitCoords
        session.save(f'clustering/probe/coords', unitCoords_)

    # Save the clustering performance
    if scores is not None:
        session.save(f'clustering/krange', krange)
        session.save(f'clustering/scores', scores)

    return krange, scores