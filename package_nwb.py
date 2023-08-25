from pynwb import TimeSeries
from pynwb.behavior import BehavioralEvents
from simply_nwb.transforms import labjack_load_file, mp4_read_data
from simply_nwb import SimpleNWB
from simply_nwb.transforms import plaintext_metadata_read
from dict_plus.utils.simpleflatten import SimpleFlattener
from simply_nwb.transforms import csv_load_dataframe_str
from pynwb.file import Subject
import pendulum
import numpy as np
import pandas as pd
import pickle
import os
import glob
from pathlib import Path
from simply_nwb.util import panda_df_to_list_of_timeseries

# Simply-NWB Package Documentation
# https://simply-nwb.readthedocs.io/en/latest/index.html

MOUSE_DETAILS = {
    "lick1": {
        "birthday": pendulum.parse("10/19/21", strict=False),
        "sex": "M",
        "strain": "C57BL/6J"  # Wild
    },
    "dcm13": {
        "birthday": pendulum.parse("7/14/22", strict=False),
        "sex": "M",
        "strain": "Gad2-Cre"
    },
    "lick3": {
        "birthday": pendulum.parse("10/19/21", strict=False),
        "sex": "M",
        "strain": "C57BL/6J"  # Wild
    },
    "lick8": {
        "birthday": pendulum.parse("7/14/22", strict=False),
        "sex": "F",
        "strain": "Gad2-Cre"  # Wild
    },

}
SESSIONS_TO_PROCESS = [
    # Folders must be in format: <prefix>/DateValue/MouseName, e.g. /media/mydrive/mydata/myfolder/2023-06-12/mouse1
    "C:\\Users\\denma\\Documents\\GitHub\\simply-nwb\\data\\2023-05-18\\lick1"
    # '/media/retina2/Seagate Expansion Drive/2023-05-18/lick1',
    # '/media/retina2/Seagate Expansion Drive/2023-05-25/lick1',
    # '/media/retina2/Seagate Expansion Drive/2023-05-26/lick1',
    # '/media/retina2/Seagate Expansion Drive/2023-05-30/lick1',
    # '/media/retina2/Seagate Expansion Drive/2023-05-18/dcm13',
    # '/media/retina2/Seagate Expansion Drive/2023-05-25/dcm13',
    # '/media/retina2/Seagate Expansion Drive/2023-05-26/dcm13',
    # '/media/retina2/Seagate Expansion Drive/2023-05-30/dcm13',
    # '/media/retina2/Seagate Expansion Drive/2023-05-18/lick3',
    # '/media/retina2/Seagate Expansion Drive/2023-05-25/lick3',
    # '/media/retina2/Seagate Expansion Drive/2023-05-26/lick3',
    # '/media/retina2/Seagate Expansion Drive/2023-05-30/lick3',
    # '/media/retina2/Seagate Expansion Drive/2023-05-18/lick8',
    # '/media/retina2/Seagate Expansion Drive/2023-05-25/lick8',
    # '/media/retina2/Seagate Expansion Drive/2023-05-26/lick8',
    # '/media/retina2/Seagate Expansion Drive/2023-05-30/lick8'
]

INSTITUTION = "CU Anschutz"

EXPERIMENTERS = [
    "Buteau, Anna"
]
LAB = "Felsen Lab"

EXPERIMENT_DESCRIPTION = "Evaluation of Perisaccadic Perceptual Changes in Mice"
EXPERIMENT_KEYWORDS = ["mouse", "saccades", "perception", "behavior", "licking"]
EXPERIMENT_RELATED_PUBLICATIONS = None

SESSION_DESCRIPTION = "Air Puff - Static Grating Head-Fixed Paradigm"

METADATA_FILENAME = "metadata.txt"

# Need multiple labjack datas?
LABJACK_FOLDER = "labjack/"
LABJACK_SUBFOLDER_GLOB = "*test*"

LABJACK_NAME = "LabjackData"
LABJACK_SAMPLING_RATE = 1000.0  # in Hz
LABJACK_DESCRIPTION = "TTL signal for when the probe, frame and airpuff is present"
LABJACK_COMMENTS = "labjack data"

MP4_FILES = {
    "RightEye": "videos/*_rightCam-0000.mp4",
    "LeftEye": "videos/*_leftCam-0000.mp4"
}
MP4_DESCRIPTION = "Camera watching the eye and and tongue"
MP4_SAMPLING_RATE = 150.0

RESPONSE_SAMPLING_RATE = MP4_SAMPLING_RATE
RESPONSE_DESCRIPTION = "description about the processed response"
RESPONSE_COMMENTS = "comments about the response"

STIM_CSVS = {
    "LeftCamStim": {
        "csv_glob": "videos/*_leftCam*.csv",
        # Units line up with
        #         bodyparts,tongue,tongue,tongue,spout,spout,spout
        "units": ["idx", "px", "px", "likelihood", "px", "px", "likelihood"]
    },
    "RightCamStim": {
        "csv_glob": "videos/*_rightCam*.csv",
        # Units line up with
        #         bodyparts,center,center,center,nasal,nasal,nasal,temporal,temporal,temporal,dorsal,dorsal,dorsal,ventral,ventral,ventral
        "units": ["idx", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood", "px", "px", "likelihood"]
    }
}

PROBE_SAMPLING_RATE = MP4_SAMPLING_RATE
PROBE_METADATA_FILE = "videos/driftingGratingWithProbeMetadata.txt"
PROBE_COMMENTS = "TODO comments about the probe here"
PROBE_DESCRIPTION = "TODO description of the probe data here"

PICKLE_FILENAME = "output.pkl"
PICKLE_DATA_NAME_PREFIX = "pickledata"
PICKLE_DATA_DESCRIPTION = "Saved data for classfied saccades, probe timestamps, puff timestamps, and frametimestamps"

PICKLE_DATA_UNITS = {
    'eyePositionUncorrected': "px",
    'eyePositionCorrected': "px",
    'eyePositionDecomposed': "px",
    'missingDataMask_left': "mask",
    'missingDataMask_right': "mask",
    'eyePositionReoriented': "px",
    'eyePositionFiltered': "px",
    'saccadeDetectionResults_waveforms_left': "px",
    'saccadeDetectionResults_waveforms_right': "px",
    'saccadeDetectionResults_indices_left': "index",
    'saccadeDetectionResults_indices_right': "index",
    'saccadeDetectionResults_amplitudes_left': "px",
    'saccadeDetectionResults_amplitudes_right': "px",
    'probeTimestamps': "s",
    'frameTimestamps': "s",
    'saccadeWaveformsLabeled_left_X': "px",
    'saccadeWaveformsLabeled_left_y': "px",
    'saccadeWaveformsLabeled_right_X': "px",
    'saccadeWaveformsLabeled_right_y': "px",
    'saccadeClassificationResults_left_nasal_indices': "index",
    'saccadeClassificationResults_left_nasal_waveforms': "px",
    'saccadeClassificationResults_left_temporal_indices': "index",
    'saccadeClassificationResults_left_temporal_waveforms': "px",
    'saccadeClassificationResults_right_nasal_indices': "index",
    'saccadeClassificationResults_right_nasal_waveforms': "px",
    'saccadeClassificationResults_right_temporal_indices': "index",
    'saccadeClassificationResults_right_temporal_waveforms': "px"
}


def run_startup_checks(session_path):
    print("Checking mp4 files..")
    for mp4_name, mp4_glob in MP4_FILES.items():
        mp4_file_glob = os.path.join(session_path, mp4_glob)
        files = glob.glob(mp4_file_glob)
        if not files:
            raise ValueError(f"Couldn't find file with glob '{mp4_file_glob}'")


def create_nwb_file(session_path):
    mouse_name = os.path.basename(session_path)  # something like 'lick1' etc
    print("Reading metadata file..")
    metadata = plaintext_metadata_read(os.path.join(session_path, METADATA_FILENAME))
    start_date = pendulum.parse(metadata["Date"], tz="local")

    if mouse_name not in MOUSE_DETAILS:
        raise ValueError(f"Unknown mouse '{mouse_name}', not found in MOUSE_DETAILS dict")

    birthday_diff = pendulum.now().diff(MOUSE_DETAILS[mouse_name]["birthday"])

    return start_date, SimpleNWB.create_nwb(
        session_description=SESSION_DESCRIPTION,
        session_start_time=start_date,
        experimenter=EXPERIMENTERS,
        subject=Subject(**{
            "subject_id": mouse_name,
            "age": f"P{birthday_diff.days}D",  # ISO-8601 for 90 days duration
            "strain": MOUSE_DETAILS[mouse_name]["strain"],
            "description": f"Mouse id '{mouse_name}'",
            "sex": MOUSE_DETAILS[mouse_name]["sex"]
        }),
        lab=LAB,
        experiment_description=EXPERIMENT_DESCRIPTION,
        session_id=mouse_name,
        institution=INSTITUTION,
        keywords=EXPERIMENT_KEYWORDS,
        related_publications=EXPERIMENT_RELATED_PUBLICATIONS
    )


def process_pickle_data(nwbfile, session_path):
    pickle_file_obj = open(os.path.join(session_path, PICKLE_FILENAME), "rb")
    pickle_data = pickle.load(pickle_file_obj)

    # Flatten the data so it's easier to add as a TimeSeries
    # e.g. {"data": {"a": [1,2,3]} }
    # gets transformed into
    # {"data_a": [1,2,3,4]}
    pickle_data = SimpleFlattener(simple_types=[np.ndarray, type(None)]).flatten(pickle_data)

    # Pop off all the None values from the pickle data
    keys_to_pop = []
    for k, v in pickle_data.items():
        if v is None:
            keys_to_pop.append(k)
    for k in keys_to_pop:
        pickle_data.pop(k)

    # Add all the data into timeseries
    timeseries_list = []
    for key, value in pickle_data.items():
        timeseries_list.append(
            TimeSeries(
                name=f"{key}",
                data=value,
                unit=PICKLE_DATA_UNITS[key],
                starting_time=0.0,
                rate=10.0,
                description=f"Measured {key}",
            ))

    # Add data to a NWBfile behavior module
    SimpleNWB.add_to_processing_module(
        nwbfile,
        module_name="behavior",
        data=BehavioralEvents(
            time_series=timeseries_list,
            name=f"BehavioralEvents"
        )
    )

    pickle_file_obj.close()


def process_labjack_data(nwbfile, session_path):
    labjack_glob = os.path.join(session_path, LABJACK_FOLDER, LABJACK_SUBFOLDER_GLOB)
    results = glob.glob(labjack_glob)
    if not results:
        raise ValueError(f"Couldn't find labjack with glob '{labjack_glob}'")
    labjack_folder = results[0]
    labjack_files = glob.glob(os.path.join(labjack_folder, "*.dat"))
    labjack_datas = []
    for labjack_file in labjack_files:
        filename = os.path.join(labjack_folder, labjack_file)
        labjack_datas.append(labjack_load_file(filename)["data"])

    labjack_combined = pd.concat(labjack_datas)

    timeseries_list = panda_df_to_list_of_timeseries(
        pd_df=labjack_combined,
        measured_unit_list=["s", "s", "s", "s", "s", "barcode", "s", "s", "s"],
        start_time=0.0,
        sampling_rate=LABJACK_SAMPLING_RATE,
        description=LABJACK_DESCRIPTION,
        comments=LABJACK_COMMENTS
    )

    SimpleNWB.add_to_processing_module(nwbfile, module_name="behavior", data=BehavioralEvents(
        time_series=timeseries_list,
        name=f"labjack_behavioral_events"
    ))


def process_mp4_data(nwbfile, session_path):
    # Add mp4 data to NWB
    for mp4_name, mp4_glob in MP4_FILES.items():
        print(f"Processing '{mp4_name}'..")
        mp4_file_glob = os.path.join(session_path, mp4_glob)
        files = glob.glob(mp4_file_glob)
        if not files:
            raise ValueError(f"Couldn't find file with glob '{mp4_file_glob}'")

        data, frames = mp4_read_data(files[0])

        SimpleNWB.mp4_add_as_acquisition(
            nwbfile,
            name=mp4_name,
            numpy_data=data,
            frame_count=frames,
            sampling_rate=MP4_SAMPLING_RATE,
            description=MP4_DESCRIPTION
        )


def process_probe_data(nwbfile, session_path):
    # PROBE CSV
    io = open(os.path.join(session_path, PROBE_METADATA_FILE))
    flines = io.readlines()
    # Fix header to csv parser works
    flines[0] = "{},Motion Value".format(",".join([r.strip() for r in flines[0].split(",")]))
    stim_df = csv_load_dataframe_str("\n".join(flines))

    # Convert dataframe to timeseries for pynwb
    probe_ts = panda_df_to_list_of_timeseries(
        stim_df,
        measured_unit_list=["index", "contrast", "motion", "value"],
        start_time=0.0,
        sampling_rate=PROBE_SAMPLING_RATE,
        description=PROBE_DESCRIPTION,
        comments=PROBE_COMMENTS
    )
    # Add data to nwbfile
    [nwbfile.add_stimulus(ts) for ts in probe_ts]


def process_response_data(nwbfile, session_path):
    # STIM_CSVS
    for name, stim_data in STIM_CSVS.items():
        csv_glob = stim_data["csv_glob"]
        fullpath = os.path.join(session_path, csv_glob)
        results = glob.glob(fullpath)
        if not results:
            raise ValueError(f"Unable to find any files matching '{fullpath}'")
        results = results[0]
        io = open(results, "r")
        lines = io.readlines()
        
        lines.pop(0)  # First line is not important, just 'scorer,<resnet name>*6'
        col_prefixes = lines.pop(0).split(",")  # Next line is the prefixes of the columns
        col_suffixes = lines.pop(0).split(",")

        # Combine the column names, insert back into data list
        headers = [f"{col_prefixes[i].strip()}_{col_suffixes[i].strip()}" for i in range(0, len(col_prefixes))]
        lines.insert(0, ",".join(headers))

        # Create the module to add the data to
        response_processing_module = nwbfile.create_processing_module(
            name=name,
            description="Processed response data for {}".format(name)
        )

        # Load CSV into a dataframe, convert to TimeSeries
        response_df = csv_load_dataframe_str("\n".join(lines))
        response_ts = panda_df_to_list_of_timeseries(
            response_df,
            measured_unit_list=stim_data["units"],
            start_time=0.0,
            sampling_rate=RESPONSE_SAMPLING_RATE,
            description=RESPONSE_DESCRIPTION,
            comments=RESPONSE_COMMENTS
        )
        # Add the timeseries into the processing module
        [response_processing_module.add(ts) for ts in response_ts]


def process_session(session_path):
    print(f"Starting session processing of '{session_path}'..")
    mouse_name = os.path.basename(session_path)  # something like 'lick1' etc

    sesspath_obj = Path(session_path)
    folderdate = pendulum.parse(sesspath_obj.parent.name, strict=False)

    print("Running startup checks..")
    run_startup_checks(session_path)

    print("Creating NWB file..")
    start_date, nwbfile = create_nwb_file(session_path)

    print("Reading labjack datas..")
    process_labjack_data(nwbfile, session_path)
    # nwbfile.processing["behavior"]["labjack_behavioral_events"]["v0"].data

    print("Reading pickle data..")
    process_pickle_data(nwbfile, session_path)
    # nwbfile.processing["behavior"]["BehavioralEvents"]["eyePositionUncorrected"].data

    print("Reading probe stim data..")
    process_probe_data(nwbfile, session_path)
    # nwbfile.stimulus["Motion"].data
    # nwbfile.stimulus["Motion Value"].data
    # nwbfile.stimulus["Motion Value"].data
    # nwbfile.stimulus["Probe contrast"].data

    print("Reading response data..")
    process_response_data(nwbfile, session_path)
    # nwbfile.processing["LeftCamStim"]["tongue_x"].data

    print("Adding MP4 Data, might take a while..")
    # process_mp4_data(nwbfile, session_path)
    # nwbfile.acquistion["LeftEyeCam"]

    print("Writing NWB file, might take a while..")
    nwbfilename = f"nwb-{folderdate.day}-{folderdate.month}-{folderdate.year}-{mouse_name}-start-{start_date.month}-{start_date.day}_{start_date.hour}-{start_date.minute}.nwb"
    print("Writing NWB '{}'..".format(nwbfilename))
    SimpleNWB.write(nwbfile, nwbfilename)
    print("Done!")


def main():
    pass


if __name__ == "__main__":
    # main()
    # for session in SESSIONS_TO_PROCESS:
        # process_session(session)
    process_session(SESSIONS_TO_PROCESS[-1])
    tw = 2