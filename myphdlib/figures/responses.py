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
        unitKeys_=None
        ):
        """
        """

        factory = SessionFactory(
            tag='JH-DATA-04'
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
        unitKeys_=None
        ):
        """
        """

        if unitKeys_ is None:
            global unitKeys
            unitKeys_ = unitKeys

        examples = self._getExampleUnits(unitKeys_)
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