from collections import defaultdict
from itertools import product


class CKSAAPFeatureExtraction:
    def __init__(self, kmax=5):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.kmax = kmax

    def calculate_cksaap_features(self, sequence):
        """
  Calculate the CKSAAP features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of CKSAAP feature values.
  """
        # Generate all possible amino acid pairs
        amino_acid_pairs = [''.join(pair) for pair in product(self.amino_acids, repeat=2)]

        # Initialize a dictionary to store the counts of each k-spaced amino acid pair
        cksaap_counts = {pair: [0] * (self.kmax + 1) for pair in amino_acid_pairs}

        # Calculate the counts of each k-spaced amino acid pair
        for k in range(self.kmax + 1):
            for i in range(len(sequence) - k - 1):
                pair = sequence[i] + sequence[i + k + 1]
                if pair in cksaap_counts:
                    cksaap_counts[pair][k] += 1

        # Calculate the window size for each k
        window_sizes = [len(sequence) - k - 1 for k in range(self.kmax + 1)]

        # Calculate the CKSAAP feature values using the formula
        cksaap_features = []
        for pair in amino_acid_pairs:
            for k in range(self.kmax + 1):
                count = cksaap_counts[pair][k]
                window_size = window_sizes[k]
                if window_size > 0:
                    cksaap_features.append(count / window_size)
                else:
                    cksaap_features.append(0)

        return cksaap_features

# # Example usage
# sequence = "ACDEFGHIKLMNPQRSTVWY"
# extractor = CKSAAPFeatureExtraction(kmax=5)
# cksaap_features = extractor.calculate_cksaap_features(sequence)
# print(cksaap_features)
