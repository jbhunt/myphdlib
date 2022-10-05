import pathlib as pl
import subprocess as sp
try:
    import cv2 as cv
except ImportError:
    cv = None

def reflectVideo(video, flipAxis='hflip', suffix=' (reflected)', speed='medium'):
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
    sp.call(command)

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