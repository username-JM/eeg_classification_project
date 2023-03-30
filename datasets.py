import numpy as np
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc
from braindecode.datautil.preprocess import NumpyPreproc
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.windowers import create_windows_from_events

from utils import print_off, print_on

def bcic4_2a(subject, low_hz=None, high_hz=None, paradigm=None, phase=False):
    X = []
    y = []

    if isinstance(subject, int):
        subject = [subject]

    for subject_id in subject:
        # Load data
        print_off()
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

        # Preprocess
        factor_new = 1e-3
        init_block_size = 1000
        preprocessors = [
            # keep only EEG sensors
            MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
            # convert from volt to microvolt
            NumpyPreproc(fn=lambda x: x * 1e+06),
            # bandpass filter
            MNEPreproc(fn='filter', l_freq=low_hz, h_freq=high_hz),
            # exponential moving standardization
            NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
                         init_block_size=init_block_size)
        ]
        preprocess(dataset, preprocessors)

        # Divide data by trial
        # - Check sampling frequency
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

        trial_start_offset_seconds = -0.5
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True
            # verbose=True
        )

        # If you need split data, try this.
        if paradigm == 'session':
            if phase == "train":
                windows_dataset = windows_dataset.split('session')['session_T']
            else:
                windows_dataset = windows_dataset.split('session')['session_E']

        # Merge subject
        for trial in windows_dataset:
            X.append(trial[0])
            y.append(trial[1])

    print_on()
    return np.array(X), np.array(y)