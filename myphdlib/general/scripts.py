import shutil
import pathlib as pl
import subprocess as sp

def syncOutputFiles(
    src='JH-DATA-04-M2',
    dst='JH-DATA-04',
    backend='shutil',
    dryrun=True,
    ):
    """
    Sync the output h5 files from one hard drive (src) to another (dst)
    """

    #
    sourceFiles = list()
    destinationFiles = list()

    # subprocess backend
    if backend == 'subprocess':
        if dryrun:
            flags = '-avrPn'
        else:
            flags = '-avrP'
        command = [
            'rsync',
            flags,
            '--dry-run',
            '--include',
            '*/',
            '--include',
            '*.hdf',
            '--exclude',
            '*',
            f'/media/jbhunt/{src}',
            f'/media/jbhunt/{dst}'
        ]
        p = sp.run(command, capture_output=True)
        for line in p.stdout.decode().split('\n'):
            if line.startswith(src) == True and line.endswith('/') == False:
                sourceFiles.append(pl.Path(line))
    
    # shutil backend
    elif backend == 'shutil':
        for date in pl.Path(f'/media/jbhunt/{src}').iterdir():
            if date.is_dir() == False:
                continue
            for animal in date.iterdir():
                sourceFile = animal.joinpath('output.hdf')
                if sourceFile.exists():
                    destinationFile = pl.Path(f'/media/jbhunt/{dst}/{date.name}/{animal.name}/output.hdf')
                    sourceFiles.append(sourceFile)
                    destinationFiles.append(destinationFile)
                    if dryrun == False:
                        shutil.copy2(
                            sourceFile,
                            destinationFile
                        )

    return sourceFiles, destinationFiles