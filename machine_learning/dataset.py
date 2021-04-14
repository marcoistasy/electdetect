from os import path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader


class Dataset:

    def __init__(self, nfff, directory_path=None, df=None):
        self.nfff = nfff  # number of frequency bands to keep
        self.directory_path = directory_path
        self.df = pd.read_csv(path.join(self.directory_path, 'segments.csv')) if df is None else df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # item for each index of the dataset

        # get the segment id as well as the category id
        segment_id, category_id = self.df.iloc[index]['segment_id'], self.df.iloc[index]['category_id']

        # load the data from the .mat file
        data = sio.loadmat(path.join(self.directory_path, segment_id + '.mat'))['data'].squeeze()

        # create spectrogram of data
        _, _, data = signal.spectrogram(data, fs=4000, noverlap=128, nfft=1024)

        # select and normalise frequency bands
        data = data[:self.nfff]
        data = preprocessing.normalize(data)
        data = np.expand_dims(data, axis=0)

        return data, category_id

    # PRIVATE

    def _split_random(self, validation_size=0.3, seed=7):
        # split the dataframe into training, validation, and evaluation sets

        # split into training dataset and validation/evaluation dataset
        train, validation_and_evaluation = train_test_split(self.df,
                                                            test_size=validation_size,
                                                            random_state=seed)

        # split validation/evaluation dataset
        validation, evaluation = train_test_split(validation_and_evaluation,
                                                  test_size=validation_size - 0.1,
                                                  random_state=seed)

        train = Dataset(self.nfff, directory_path=self.directory_path, df=train)
        validation = Dataset(self.nfff, directory_path=self.directory_path, df=validation)
        evaluation = Dataset(self.nfff, directory_path=self.directory_path, df=evaluation)

        # return the datasets
        return train, validation, evaluation

    # PUBLIC

    def create_data_loaders(self, validation_size, batch_size, nworkers, seed, replacement):
        training_dataset, validation_dataset, evaluation_dataset = self._split_random(validation_size, seed)

        # create data sampler to rectify class imbalance
        sampler = Dataset._construct_weighted_sampler(training_dataset.df, 'category_id', replacement)

        # create parameters for data loader
        params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': nworkers, 'drop_last': False}

        # create data loader objects
        train = DataLoader(training_dataset, sampler=sampler, **params)
        valid = DataLoader(validation_dataset, **params)
        evaluate = DataLoader(evaluation_dataset, **params)

        return train, valid, evaluate

    @staticmethod
    def _construct_weighted_sampler(df, class_column_id, replacement):
        # draw samples from multinomial distribution given weights

        # get unique labels and counts from a data frame
        labels_unique, counts = np.unique(df[class_column_id], return_counts=True)

        # produce weights and apply them to each class
        class_weights = [sum(counts) / c for c in counts]
        example_weights = [class_weights[e] for e in df[class_column_id]]

        # produce and return sampler
        return WeightedRandomSampler(example_weights, len(df[class_column_id]), replacement=replacement)
