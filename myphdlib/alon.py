import shutil
import numpy as np
import pathlib as pl
from myphdlib.pipeline import iterateSessions

def createShareableDataset(datasetName='Realtime', workingDirectory='/home/jbhunt/Desktop/', datasetSuffix=None, overwrite=True):
    """
    """

    #
    workingDirectoryPath = pl.Path(workingDirectory)
    if datasetSuffix is not None:
        destinationFolderPath = workingDirectoryPath.joinpath(datasetName + str(datasetSuffix))
    else:
        destinationFolderPath = workingDirectoryPath.joinpath(datasetName)

    #
    if destinationFolderPath.exists():
        if overwrite:
            shutil.rmtree(str(destinationFolderPath))
        else:
            raise Exception('Destination folder already exists: {destinationFolderPath}')

    #
    for obj, animal, date, session in iterateSessions(datasetName):

        if obj.isValid() == False:
            continue
        
        # Copy the directory structure
        try:
            if session != None:
                animal += session
            sessionFolderPath = destinationFolderPath.joinpath(date).joinpath(animal)
            sessionFolderPath.mkdir(parents=True)

            #
            frameCount = obj.eyePositionDecomposed.shape[0]
            frameOnsetTimestamps = obj.load('labjackTimestampsAcquisition')[:frameCount]
            np.savetxt(str(sessionFolderPath.joinpath('frameOnsetTimestamps.txt')), frameOnsetTimestamps, fmt='%.3f')

            #
            if hasattr(obj, 'probeOnsetTimestamps'):
                probeOnsetTimestamps = obj.labjackData[obj.probeOnsetIndices, 0]
                np.savetxt(str(sessionFolderPath.joinpath('probeOnsetTimestamps.txt')), probeOnsetTimestamps, fmt='%.3f')

            try:
                motionOnsetTimestamps = obj.load('motionOnsetTimestamps')
                motionOffsetTimestamps = obj.load('motionOffsetTimestamps')
                np.savetxt(str(sessionFolderPath.joinpath('motionOnsetTimestamps.txt')), motionOnsetTimestamps, fmt='%.3f')
                np.savetxt(str(sessionFolderPath.joinpath('motionOffsetTimestamps.txt')), motionOffsetTimestamps, fmt='%.3f')
            except:
                pass


            try:
                lightOnsetTimestamps = obj.load('lightOnsetTimestamps')
                lightOffsetTimestamps = obj.load('lightOffsetTimestamps')
                np.savetxt(str(sessionFolderPath.joinpath('lightOnsetTimestamps.txt')), lightOnsetTimestamps, fmt='%.3f')
                np.savetxt(str(sessionFolderPath.joinpath('lightOffsetTimestamps.txt')), lightOffsetTimestamps, fmt='%.3f')
            except:
                pass

            #
            searchResults = list(obj.sessionFolderPath.glob('*Metadata*'))
            if len(searchResults) == 1:
                stimulusMetadataPath = searchResults.pop()
                shutil.copy(str(stimulusMetadataPath), str(sessionFolderPath.joinpath(stimulusMetadataPath.name)))

            #
            saccadeOnsetTimestamps = list()
            for targetEye in ('left', 'right'):
                for saccadeDirection in ('left', 'right'):
                    saccadeOnsetIndices = obj.saccadeOnsetIndicesClassified[targetEye][saccadeDirection]
                    for saccadeOnsetTimestamp in frameOnsetTimestamps[saccadeOnsetIndices]:
                        entry = [
                            -1 if targetEye == 'left' else 1,
                            -1 if saccadeDirection == 'left' else 1,
                            saccadeOnsetTimestamp
                        ]
                        saccadeOnsetTimestamps.append(entry)
            saccadeOnsetTimestamps = np.array(saccadeOnsetTimestamps, dtype=object)
            timeSortedIndex = np.argsort(saccadeOnsetTimestamps[:, -1])
            np.savetxt(
                str(sessionFolderPath.joinpath(f'saccadeOnsetTimestamps.txt')),
                saccadeOnsetTimestamps[timeSortedIndex, :],
                fmt=('%.0f', '%.0f', '%.3f'),
                delimiter=', ',
                header='Eye (-1=Left, +1=Right), Direction (-1=Left, +1=Right), Timestamp',
                comments=''
            )

            #
            np.savetxt(str(sessionFolderPath.joinpath('eyePositionDecomposed.txt')), obj.eyePositionDecomposed, fmt='%.3f', delimiter=', ')
            np.savetxt(str(sessionFolderPath.joinpath('eyePositionStandardized.txt')), obj.eyePositionStandardized, fmt='%.3f', delimiter=', ')
            np.savetxt(str(sessionFolderPath.joinpath('eyePositionCorrected.txt')), obj.eyePositionCorrected, fmt='%.3f', delimiter=', ')

        except Exception as error:
            print(error)
            shutil.rmtree(str(sessionFolderPath))

    return