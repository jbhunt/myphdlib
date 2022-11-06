from myphdlib.general.session import saveSessionData, locateFactorySource

class Session():
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        # Folders
        self.sessionFolderPath = pl.Path(sessionFolder)
        self.videosFolderPath = self.sessionFolderPath.joinpath('videos')

        # Files
        self.inputFilePath = self.sessionFolderPath.joinpath('input.txt')
        self.outputFilePath = self.sessionFolderPath.joinpath('output.pickle')
        self.driftingGratingMetadataFilePath = self.videosFolderPath.joinpath('driftingGratingMetadata.txt')

        # Determine the animal, date, and treatment
        self.notesFilePath = self.sessionFolderPath.joinpath('notes.txt')
        self.animal, self.date, self.treatment = None, None, None
        if self.notesFilePath.exists():
            with open(self.notesFilePath, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                for attribute in ('animal', 'date', 'experiment'):
                    if bool(re.search(f'{attribute}*', line.lower())) and line.startswith('-') == False:
                        value = line.lower().split(': ')[-1].rstrip('\n')
                        setattr(self, attribute, value)

        return

    def load(self, name):
        """
        """

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            dataContainer = pickle.load(stream)

        if name not in dataContainer.keys():
            raise Exception(f'Invalid data key: {name}')
        else:
            return dataContainer[name]

    @property
    def fps(self):
        """
        Video acquisition framerate
        """

        framerate = None
        result = list(self.videosFolderPath.glob('*metadata.yaml'))
        if result:
            with open(result.pop(), 'r') as stream:
                acquisitionMetadata = yaml.safe_load(stream)
            for cameraAlias in ('cam1', 'cam2'):
                if acquisitionMetadata[cameraAlias]['ismaster']:
                    framerate = acquisitionMetadata[cameraAlias]['framerate']

        return framerate

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

        sessionLocated = False
        for session in self:
            if session.animal == animal and session.date == date:
                sessionLocated = True
                break
        
        if sessionLocated:
            return session
        else:
            raise Exception('Could not locate session')

    # Iterator protocol definition
    def __iter__(self):
        self.sessionFolders = list()
        for date in self.rootFolderPath.iterdir():
            for animal in date.iterdir():
                self.sessionFolders.append(str(animal))
        self._listIndex = 0
        return self

    def __next__(self):
        if self._listIndex < len(self.sessionFolders):
            sessionFolder = self.sessionFolders[self._listIndex]
            self._listIndex += 1
            return Session(sessionFolder)
        else:
            raise StopIteration