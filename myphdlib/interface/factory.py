import os
import re
import yaml
if os.name == 'nt':
    import win32api
import pathlib as pl
#from datetime import dated
from string import ascii_uppercase as letters
from myphdlib.interface.muscimol import MuscimolSession

class SessionFactory(object):
    """
    """

    def __init__(
        self,
        tag='JH-DATA-'
        ):
        """
        """

        self._findDataVolumes(tag)
        # self._loadExperimentData()

        return
    
    def _findDataVolumes(
        self,
        tag='JH-DATA-'
        ):
        """
        """

        self.volumes = list()

        # Windows
        if os.name == 'nt':
            for letter in letters:
                drive = pl.Path(f'{letter}:/')
                if drive.exists() == False:
                    continue
                name, serialno, mcl, flags, system = win32api.GetVolumeInformation(str(drive))
                if bool(re.search(f'.*{tag}.*', name)):
                    self.volumes.append(drive)

        # Linux
        elif os.name == 'posix':
            for user in pl.Path('/media/').iterdir():
                for drive in user.iterdir():
                    if bool(re.search(f'.*{tag}.*', str(drive.name))):
                        self.volumes.append(drive)

        #
        else:
            raise Exception(f'{os.name} is not a supported OS')
        
        return
    
    def _loadExperimentData(
        self,
        ):
        """
        """

        self.metadata = None
        for volume in self.volumes:
            file = volume.joinpath('experiments.yml')
            if file.exists():
                with open(file, 'r') as stream:
                    self.metadata = yaml.full_load(stream)
                    return

        #
        if self.metadata is None:
            raise Exception('Could not locate experiments metadata file')
        
        return
    
    def produce(
        self,
        experiment=None,
        cohort=None,
        animals=None,
        dates=(None, None),
        letters=(None, 'a', 'b', 'c'),
        ):
        """
        """

        keys = list()
        for experiment_ in self.metadata.keys():
            if experiment is not None and experiment_ != experiment:
                continue
            for cohort_ in self.metadata[experiment_].keys():
                if cohort is not None and int(cohort_[-1]) != cohort:
                    continue
                for animal_ in self.metadata[experiment_][cohort_].keys():
                    if animals is not None and animal_ not in animals:
                        continue
                    for date_ in self.metadata[experiment_][cohort_][animal_]:
                        if dates[0] is not None:
                            if date_ < date.fromisoformat(dates[0]):
                                continue 
                        if dates[1] is not None:
                            if date_ > date.fromisoformat(dates[1]):
                                continue
                        for letter in letters:
                            entry = (
                                experiment_,
                                cohort_,
                                animal_ if letter is None else animal_ + letter,
                                date_
                            )
                            keys.append(entry)

        #
        sessions = list()
        for experiment_, cohort_, animal_, date_ in keys:
            for volume in self.volumes:
                folder = volume.joinpath(str(date_), animal_)
                if folder.exists():
                    if experiment_ == 'Muscimol':
                        session = MuscimolSession(folder)
                    sessions.append(session)

        return sessions