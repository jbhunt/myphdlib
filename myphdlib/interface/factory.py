import os
import re
import yaml
if os.name == 'nt':
    import win32api
import pathlib as pl
from datetime import date
from string import ascii_uppercase as letters
from myphdlib.interface.muscimol import MuscimolSession
from myphdlib.interface.suppression import SuppressionSession
from myphdlib.interface.mlati import MlatiSession

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
        self._loadExperimentData()

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

        #
        if type(animals) == str:
            animals = (animals,)
        if type(dates) == str:
            dates = (dates,)

        keys = list()
        for experiment_ in self.metadata.keys():
            if experiment is not None and experiment_ != experiment:
                continue
            for animal_ in self.metadata[experiment_].keys():
                if animals is not None and animal_ not in animals:
                    continue
                for date_ in self.metadata[experiment_][animal_]:

                    # Single date
                    if len(dates) == 1:
                        if date_ != date.fromisoformat(dates[0]):
                            continue

                    # Date range
                    if len(dates) == 2:
                        if dates[0] is not None:
                            if date_ < date.fromisoformat(dates[0]):
                                continue 
                        if dates[1] is not None:
                            if date_ > date.fromisoformat(dates[1]):
                                continue

                    # List of dates
                    if len(dates) > 2:
                        if str(date_) not in dates:
                            continue

                    #
                    for letter in letters:
                        entry = (
                            experiment_,
                            animal_ if letter is None else animal_ + letter,
                            date_
                        )
                        keys.append(entry)

        #
        sessions = list()
        for experiment_, animal_, date_ in keys:
            for volume in self.volumes:
                folder = volume.joinpath(str(date_), animal_)
                if folder.exists():
                    if experiment_ == 'Muscimol':
                        session = MuscimolSession(folder)
                    elif experiment_ == 'Suppression':
                        session = SuppressionSession(folder)
                    elif experiment_ == 'Mlati':
                        session = MlatiSession(folder)
                    else:
                        continue
                    sessions.append(session)

        #
        filtered = list()
        for session in sessions:
            if cohort is None:
                filtered.append(session)
            else:
                if session.cohort != cohort:
                    continue
                filtered.append(session)

        return filtered