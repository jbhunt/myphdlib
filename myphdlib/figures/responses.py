import numpy as np
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import psth2
from myphdlib.interface.factory import SessionFactory

unitKeys = (
    ('mlati9', '2023-07-10', 560),
    ('mlati6', '2023-04-14', 577),
    ('mlati7', '2023-05-15', 436),
    ('mlati10', '2023-07-19', 380),
    ('mlati7', '2023-05-15', 404)
)

class ExampleVisualNeuronsFigure():
    """
    """

    def __init__(self):
        """
        """

        self.data = None

        return

    def _getExampleUnits(
        self,
        unitKeys_=None,
        remote=False,
        ):
        """
        """

        if remote:
            factory = SessionFactory(
                mount='/home/josh/mygdrive/Josh'
            )
        else:
            factory = SessionFactory(
                tag='JH-DATA-04',
            )
        examples = list()

        for animal, date, cluster in unitKeys_:
            session = factory.produce(
                animals=(animal,),
                dates=(date,)
            ).pop()
            unit = session.population.indexByCluster(cluster)
            examples.append(unit)

        return examples

    def generate(
        self,
        window=(-0.3, 0.5),
        figsize=(1.75, 6.3),
        unitKeys_=None,
        remote=False,
        ):
        """
        """

        if unitKeys_ is None:
            global unitKeys
            unitKeys_ = unitKeys

        examples = self._getExampleUnits(unitKeys_, remote=remote)
        nUnits = len(examples)

        fig, axs = plt.subplots(nrows=nUnits, sharex=True)
        if nUnits == 1:
            axs = (axs,)
        for unit, ax in zip(examples, axs):
            t, M = psth2(
                unit.session.probeTimestamps,
                unit.timestamps,
                window=window,
                binsize=0.02,
            )
            fr = M.mean(0) / 0.02
            ax.plot(t, fr, color='k')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig

class VisualResponseSummaryFigure():
    """
    """

    def __init__(self):
        """
        """

        self.data = None

        return
    
    def generate(
        self,
        sessions,
        visualResponseAmplitude=20,
        nColumns=5,
        vrange=(-3, 3),
        ):
        """
        """

        R = list()
        for session in sessions:
            if session.probeTimestamps is None:
                continue
            session.population.filter(
                reload=True,
                visualResponseAmplitude=None
            )
            for unit in session.population:
                t, z = unit.peth(
                    session.probeTimestamps,
                    responseWindow=(-0.3, 0.5),
                    binsize=0.02,
                    standardize=True
                )
                if np.isnan(z).all():
                    continue
                R.append(z)

        #
        R = np.array(R)
        self.data = R

        #
        fig, axs = plt.subplots(ncols=nColumns, sharex=True, sharey=True)
        nUnitsPerSubplot = int(np.ceil(R.shape[0] / nColumns))
        splits = np.split(R, np.arange(nUnitsPerSubplot, R.shape[0], nUnitsPerSubplot))

        for r, ax in zip(splits, axs):
            ax.pcolor(r, vmin=vrange[0], vmax=vrange[1], rasterized=True, cmap='binary_r')
        fig.tight_layout()

        return fig