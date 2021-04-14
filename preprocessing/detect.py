# %%

# imports

import mne
from mne_bids import read_raw_bids, BIDSPath

# constants

SUBJECT_IDs = [0, 1, 2, 3, 4, 5]
INPUT_DIRECTORY = ''

for subject_id in SUBJECT_IDs:

    # get bids path
    bids_path = BIDSPath(subject=str(subject_id),
                         session='01',
                         task='continuous',
                         run='01',
                         root=INPUT_DIRECTORY)

    # load data
    raw = read_raw_bids(bids_path=bids_path)

    # NOTE: THERE ARE TWO WAYS TO EPOCH THE DATA. THE FIRST IS TO USE THE ANNOTATIONS WHICH ALREADY EXIST

    # create events and epochs
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id)

    # %%
    epochs['EOG'].average().plot()
    epochs['ECG'].average().plot()

    # THE SECOND IS TO CREATE THE EPOCHS FROM THE ECG AND EOG LEADS (THIS PART IS COMMENTED BELOW)

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=-1.5, tmax=1.5, baseline=(-0.5, -0.2))
    # eog_epochs = mne.preprocessing.create_eog_epochs(raw, tmin=-1.5, tmax=1.5, baseline=(-0.5, -0.2))
