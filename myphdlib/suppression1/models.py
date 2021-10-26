import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from skimage.measure import EllipseModel
from matplotlib import pylab as plt
from matplotlib.patches import Ellipse
from scipy.stats import pearsonr

from .  import helpers
from .. import toolkit

class ModelVisualNeuron():
    """
    """

    def __init__(self, session, unit):
        """
        """

        self._session = session
        self._unit = unit
        self._trained = False
        self._model = None

        return

    def fit(
        self,
        colliculus='left',
        motor_response_window=(0, 0.2),
        peristimulus_exclusion_window=(-0.5, 0.5),
        hanning_window_length=7,
        plot=True,
        **psth_kwargs_
        ):
        """
        """

        #
        psth_kwargs = {
            'binsize': 0.01,
            'window' : (-0.01, 0.3)
        }
        psth_kwargs.update(psth_kwargs_)

        #
        eye = 'left' if colliculus == 'right' else 'right'

        # Mask for identifying the visual response
        psth_edges = np.arange(
            psth_kwargs['window'][0],
            psth_kwargs['window'][1] + psth_kwargs['binsize'],
            psth_kwargs['binsize']
        )
        response_window_mask = np.logical_and(
            psth_edges[:-1] + psth_kwargs['binsize'] / 2 >= motor_response_window[0],
            psth_edges[:-1] + psth_kwargs['binsize'] / 2 <= motor_response_window[1]
        )

        #
        waveforms = self._session.binocular_saccade_waveforms

        #
        X_peristimulus, y_peristimulus = list(), list()
        X_extrastimulus, y_extrastimulus = list(), list()
        for direction, saccade_onset_timestamps in self._session.saccade_onset_timestamps.items():

            # Create a mask that excludes saccades that coincide with probes
            peristimulus = helpers.create_coincidence_mask(
                saccade_onset_timestamps,
                np.concatenate([
                    self._session.probe_onset_timestamps['low'],
                    self._session.probe_onset_timestamps['medium'],
                    self._session.probe_onset_timestamps['high'],
                ]),
                window=peristimulus_exclusion_window
            )
            extrastimulus = np.invert(peristimulus)

            # Collect the peri-stimulus responses
            for irow, saccade_onset_timestamp in enumerate(saccade_onset_timestamps[peristimulus]):

                #
                t, M = toolkit.psth(
                    [saccade_onset_timestamp],
                    self._unit.timestamps,
                    **psth_kwargs
                )
                # response = toolkit.smooth(M.mean(0), hanning_window_length)[response_window_mask].sum()
                response = M.mean(0)[response_window_mask].sum()
                y_peristimulus.append(response)

                #
                waveform = waveforms[direction][eye][irow]
                X_peristimulus.append(waveform)

            # Collect the extra-stimulus responses
            for irow, saccade_onset_timestamp in enumerate(saccade_onset_timestamps[extrastimulus]):

                #
                t, M = toolkit.psth(
                    [saccade_onset_timestamp],
                    self._unit.timestamps,
                    **psth_kwargs
                )
                # response = toolkit.smooth(M.mean(0), hanning_window_length)[response_window_mask].sum()
                response = M.mean(0)[response_window_mask].sum()
                y_extrastimulus.append(response)

                #
                waveform = waveforms[direction][eye][irow]
                X_extrastimulus.append(waveform)

        # Convert to numpy arrays
        X_peristimulus = np.array(X_peristimulus)
        y_peristimulus = np.array(y_peristimulus)
        X_extrastimulus = np.array(X_extrastimulus)
        y_extrastimulus = np.array(y_extrastimulus)

        # Split extra-stimulus responses into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X_extrastimulus, y_extrastimulus, test_size=0.5)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # model = LinearRegression(fit_intercept=False)
        model = MLPRegressor(max_iter=10000, tol=0.0001)
        # model = Ridge()
        self._model = model.fit(X_train, y_train)
        self._trained = True

        if plot:
            fig, ax = plt.subplots()
            ymax = 0
            iterable = zip(
                ['r', 'b', 'g'],
                [X_peristimulus, X_train, X_test],
                [y_peristimulus, y_train, y_test],
                [r'peristimulus', r'extrastimulus (train)', r'extrastimulus (test)']
            )
            for color, X, y_true, label in iterable:

                #
                y_prediction = self._model.predict(X)

                #
                r, p = pearsonr(y_prediction, y_true)
                r2 = np.around(r ** 2, 3)
                whole_label = label + r' ($r^2$' + f'={r2:.3f})'

                #
                ax.scatter(y_prediction, y_true, color=color, alpha=0.15, marker='o', label=whole_label, s=10)

                # Fit an ellipse
                ellipse = EllipseModel()
                ellipse.estimate(np.hstack([y_prediction.reshape(-1, 1), y_true.reshape(-1, 1)]))
                xc, yc, a, b, theta = ellipse.params
                patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor=color, facecolor='none', alpha=0.5)
                ax.add_patch(patch)

                #
                if y_prediction.max() > ymax:
                    ymax = y_prediction.max()
                if y_true.max() > ymax:
                    ymax = y_true.max()

            ax.legend()
            ax.plot([0, ymax], [0, ymax], color='k', linestyle='--')
            ax.set_xlim([-0.5, ymax + 0.5])
            ax.set_ylim([-0.5, ymax + 0.5])
            fig.set_figwidth(5)
            fig.set_figheight(5)

            return

        return X_train, y_train, X_peristimulus, y_peristimulus

    def predict(self, X):
        """
        """

        if self._trained is False:
            raise Exception('Model neuron is not trained')

        return self._model.predict(X)
