import re
import pathlib as pl
from . import ffmpeg

def findAndCompressRawVideos(rootFolder, dryRun=False, deleteRawVideos=False):
    """
    Recursively search for raw (uncompressed) videos, compress them,
    and delete the original files
    """

    #
    todoList = list()

    #
    for folder in pl.Path(rootFolder).rglob('*'):

        #
        if folder.is_dir() == False:
            continue

        #
        if bool(re.search('session*', folder.name)):

            # See what videos already exist in the session folder
            videos = {
                'mp4': list(),
                'avi': list()
            }
            for file in folder.iterdir():
                if file.suffix in ('.avi', '.mp4'):
                    videos[file.suffix.strip('.')].append(file)
            
            # See if each avi has a corresponding mp4
            for aviFilePath in videos['avi']:

                #
                flag = False
                for mp4FilePath in videos['mp4']:
                    if mp4FilePath.stem == aviFilePath.stem:
                        flag = True
                        break
                
                #
                if flag == False:
                    todoList.append(aviFilePath)
                    
    #
    if dryRun:
        return [str(filePath) for filePath in todoList]

    else:
        failed = list()
        for videoFilePath in todoList:
            print(f'Working on {videoFilePath.name} ...')
            checkPassed = ffmpeg.compressVideo(str(videoFilePath))
            if checkPassed == False:
                failed.append(str(videoFilePath))
            else:
                if deleteRawVideos:
                    videoFilePath.unlink()

        #
        if len(failed) != 0:
            print('Compression failed for the following videos:')
            for videoFileName in failed:
                print(f'- {videoFileName}')

    return