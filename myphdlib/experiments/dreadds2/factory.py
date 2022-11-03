from myphdlib.general.session import saveSessionData, locateFactorySource

class Session():
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        return

def SessionFactory(SessionFactoryBase):
    """
    """

    def __init__(self, hdd='CM-DATA-00', alias='Dreadds2', source=None):
        """
        """

        kwargs = {
            'hdd': hdd,
            'alias': alias,
            'source': source
        }
        self.rootFolderPath = locateFactorySource(**kwargs)
        self.sessionFolders = list()

        return

    def produce(self, animal, date):
        """
        """

        return

    def __iter__(self):
        return

    def __next__(self):
        return