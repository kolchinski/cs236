import os
import csv
import numpy as np


class NIPS2015Dataset(object):
    def __init__(self, batch_size=1, seq_len=25, data_folder='data/'):
        """
        Initialize the data iterator of NIPS 2015 papers.
        :param batch_size [int]: The batch size for each mini-batch
        :param seq_len [int]: Each string in the mini-batch will be truncated to length `seq_len`
        :param data_folder [string]: The directory that stores the `paper.csv` file.
        """
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.papers = []
        if os.path.exists(data_folder):
            with open(os.path.join(data_folder, 'papers.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.papers.append(row['PaperText'])
                self.all_texts = '\n'.join(self.papers)
        else:
            raise FileNotFoundError("Please download papers.csv")

        self.vocabulary = sorted(list(set(self.all_texts)))
        self.voc_len = len(self.vocabulary)
        self.idx2char = {i: c for i, c in enumerate(self.vocabulary)}
        self.char2idx = {c: i for i, c in enumerate(self.vocabulary)}
        self.voc_freq = np.zeros(len(self.vocabulary))
        for c in self.all_texts:
            self.voc_freq[self.char2idx[c]] += 1
        self.voc_freq /= np.sum(self.voc_freq)

        assert batch_size <= len(self.papers), "The batch size should not be larger than the number of papers"

        self.texts = [list() for _ in range(batch_size)]
        for idx in range(len(self.papers)):
            self.texts[idx % batch_size].append(self.papers[idx])
        for batch in range(batch_size):
            self.texts[batch] = '\n'.join(self.texts[batch])
        self.batch_len = min(len(self.texts[batch]) for batch in range(batch_size))

        assert self.batch_len >= seq_len + 1, "Two many batches for such a sequence length"

        self.p = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        """
        :return [(np.ndarray, np.ndarray)]: Two numpy arrays each representing B strings of length seq_len.
        Both have the shape B x seq_len. The second array is shifted from the first array by one time step.
        """
        if self.p + self.seq_len + 1 >= self.batch_len:
            raise StopIteration
        batch_input = [[self.char2idx[self.texts[batch][i]] for i in range(self.p, self.p + self.seq_len + 1)]
                       for batch in range(self.batch_size)]
        self.p += self.seq_len
        return np.array(batch_input)[:, :-1].astype(np.int32), \
                np.array(batch_input)[:, 1:].astype(np.int32)

    def _reset(self):
        self.p = 0
