import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt


class PairsParser:

    def getSupportedExtensions(self):
        return list()

    def __parse_pairs__(self, filepath, delimiter=',', target_col=2, column_names=list(), sequence_length=None):
        assert("target" in column_names)
        with open(filepath, "r") as f:
            lines = f.readlines()
            try:
                if sequence_length is None:
                    dataframe = pd.read_csv(filepath, sep=delimiter, skip_blank_lines=True,
                        header=None, names=column_names, index_col=False)
                    sequence_length = np.asarray(dataframe[["i", "j"]]).max()
            except ValueError:
                return None
            data = np.full((sequence_length, sequence_length), np.nan, dtype=np.double)
            np.fill_diagonal(data, 0)
            for line in lines:
                elements = line.rstrip("\r\n").split(delimiter)
                i, j, k = int(elements[0]) - 1, int(elements[1]) - 1, float(elements[target_col])
                data[i, j] = data[j, i] = k
            if np.isnan(data).any():
                # sequence_length is wrong or the input file has missing pairs
                warnings.warn("Warning: Pairs of residues are missing from the contacts text file")
                warnings.warn("Number of missing pairs: %i " % np.isnan(data).sum())
            return data


class CBParser(PairsParser):

    def getSupportedExtensions(self):
        return [".contacts", ".CB"]

    def parse(self, filepath):
        return self.__parse_pairs__(filepath, delimiter=' ', target_col=2,
            column_names = ["i", "j", "target"])


distances = np.squeeze(CBParser().parse('closest.Jan13.contacts'))

for tau in [5.0, 6.5, 8.0, 9.5, 11.0]:
    cmap = 1. / (1. + np.exp(0.8 * (distances - tau)))

    plt.imshow(cmap, cmap='gist_earth')
    plt.axis('off')
    plt.savefig('cmap%f.png' % tau)
