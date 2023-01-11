import os
import pathlib as pl
import subprocess as sp
try:
    import cv2 as cv
except ImportError:
    cv = None

def ffmpegInstalled():
    """
    """

    return

def reflectVideo(video, flipAxis='hflip', suffix=' (reflected)', speed='medium', deleteOriginalVideo=False):
    """
    """

    videoFilePath = pl.Path(video)
    workingDirectory = videoFilePath.parent
    filename = videoFilePath.name.rstrip('.mp4')
    reflectedVideoFilePath = workingDirectory.joinpath(videoFilePath.parent, f'{filename}{suffix}.mp4')

    command = [
        'ffmpeg',
        '-i', str(videoFilePath),
        '-vf', flipAxis,
        '-preset', speed,
        '-c:a', 'copy',
        reflectedVideoFilePath
    ]
    if os.name == 'nt':
        sp.call(command, shell=True)
    else:
        sp.call(command)

    #
    if deleteOriginalVideo:
        checkPassed = False
        try:
            frameCountOriginal = countVideoFrames(str(videoFilePath), backend='OpenCV')
            frameCountPost = countVideoFrames(str(reflectedVideoFilePath), backend='OpenCV')
            if frameCountOriginal == frameCountPost:
                checkPassed = True

        except Exception as error:
            checkPassed = False
        
        if checkPassed:
            videoFilePath.unlink()

    return

def compressVideo(video, preset='veryfast', crf=17):
    """
    """

    sourceVideoPath = pl.Path(video)
    filename = sourceVideoPath.name.rstrip(sourceVideoPath.suffix)
    destinationVideoPath = sourceVideoPath.parent.joinpath(f'{filename}.mp4')
    if destinationVideoPath.exists():
        raise Exception('Compressed video already exists')

    command = [
        'ffmpeg',
        '-y',
        '-i', str(sourceVideoPath),
        '-c:v', 'libx265',
        '-preset', preset,
        '-strict', 'experimental',
        '-crf', str(crf),
        '-loglevel', 'quiet',
        str(destinationVideoPath)
    ]

    #
    checkPassed = False
    try:
        sp.call(command)
        frameCountRaw = countVideoFrames(str(sourceVideoPath), backend='OpenCV')
        frameCountCompressed = countVideoFrames(str(destinationVideoPath))
        if frameCountRaw == frameCountCompressed:
            checkPassed = True
    
    except:
        checkPassed = False

    #
    if checkPassed == False and destinationVideoPath.exists():
        destinationVideoPath.unlink()

    return checkPassed

def countVideoFrames(video, backend='ffprobe'):
    """
    """

    if backend == 'ffprobe':
        args = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries',
            'stream=nb_read_packets',
            '-of',
            'csv=p=0',
            video
        ]
        process = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
        result, error = process.communicate()
        result = int(result)
    
    elif backend == 'OpenCV':
        if cv != None:
            cap = cv.VideoCapture(video)
            result = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    return result