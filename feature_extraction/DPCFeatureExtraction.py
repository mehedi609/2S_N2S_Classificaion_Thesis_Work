from collections import Counter
from itertools import product


class DPCFeatureExtraction:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    def calculate_dpc_features(self, sequence):
        """
        Calculate the DPC features for a given protein sequence.

        :param sequence: The protein sequence as a string.
        :return: A list of DPC feature values.
        """
        # Generate all possible dipeptide combinations
        dipeptides = [''.join(dipeptide) for dipeptide in product(self.amino_acids, repeat=2)]

        # Initialize a dictionary to store the counts of each dipeptide
        dipeptide_counts = {dipeptide: 0 for dipeptide in dipeptides}

        # Count the occurrences of each dipeptide in the sequence
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            if dipeptide in dipeptide_counts:
                dipeptide_counts[dipeptide] += 1

        # Calculate the sequence length
        sequence_length = len(sequence)

        # Calculate the DPC feature values using the formula
        dpc_features = [dipeptide_counts[dipeptide] / (sequence_length - 1) for dipeptide in dipeptides]

        return dpc_features
