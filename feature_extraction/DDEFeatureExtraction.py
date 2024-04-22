from collections import Counter
from itertools import product
import math


class DDEFeatureExtraction:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.codons = {
            'A': 4, 'R': 6, 'N': 2, 'D': 2, 'C': 2,
            'Q': 2, 'E': 2, 'G': 4, 'H': 2, 'I': 3,
            'L': 6, 'K': 2, 'M': 1, 'F': 2, 'P': 4,
            'S': 6, 'T': 4, 'W': 1, 'Y': 2, 'V': 4
        }
        self.total_codons = sum(self.codons.values())

    def calculate_dde_features(self, sequence):
        """
        Calculate the DDE features for a given protein sequence.

        :param sequence: The protein sequence as a string.
        :return: A list of DDE feature values.
        """
        dpc_features = self.calculate_dpc_features(sequence)
        dde_features = []

        for x, y in product(self.amino_acids, repeat=2):
            dipeptide = x + y
            dpc = dpc_features.get(dipeptide, 0)  # Use get() to handle missing dipeptides
            tm = self.calculate_tm(x, y)
            tv = self.calculate_tv(tm, len(sequence))
            dde = self.calculate_dde(dpc, tm, tv)
            dde_features.append(dde)

        return dde_features

    def calculate_dpc_features(self, sequence):
        """
        Calculate the dipeptide composition (DPC) features.

        :param sequence: The protein sequence as a string.
        :return: A dictionary of DPC feature values.
        """
        possible_dipeptides = [''.join(dp) for dp in product(self.amino_acids, repeat=2)]
        dipeptide_counts = Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))
        total_dipeptides = sum(dipeptide_counts.values())
        dpc = {dp: dipeptide_counts[dp] / total_dipeptides for dp in possible_dipeptides}
        return dpc

    def calculate_tm(self, x, y):
        """
        Calculate the theoretical mean (Tm) for amino acids x and y.

        :param x: The first amino acid.
        :param y: The second amino acid.
        :return: The theoretical mean value.
        """
        cx = self.codons[x]
        cy = self.codons[y]
        tm = (cx / self.total_codons) * (cy / self.total_codons)
        return tm

    def calculate_tv(self, tm, sequence_length):
        """
        Calculate the theoretical variance (Tv) based on Tm and sequence length.

        :param tm: The theoretical mean value.
        :param sequence_length: The length of the protein sequence.
        :return: The theoretical variance value.
        """
        tv = (tm * (1 - tm)) / (sequence_length - 1)
        return tv

    def calculate_dde(self, dpc, tm, tv):
        """
        Calculate the deviation from expected dipeptide composition (DDE).

        :param dpc: The dipeptide composition value.
        :param tm: The theoretical mean value.
        :param tv: The theoretical variance value.
        :return: The DDE value.
        """
        if tv == 0:
            return 0
        else:
            dde = (dpc - tm) / math.sqrt(tv)
            return dde
