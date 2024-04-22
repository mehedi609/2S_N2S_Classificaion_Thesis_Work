from collections import Counter
from itertools import product


class GTPCFeatureExtraction:
    def __init__(self):
        self.amino_acid_groups = {
            'g1': 'FYW',  # Aromatic
            'g2': 'RKH',  # Positively charged
            'g3': 'GAVLMI',  # Aliphatic
            'g4': 'STCPNQ',  # Uncharged
            'g5': 'DE'  # Negatively charged
        }
        self.group_names = list(self.amino_acid_groups.keys())

    def calculate_gtpc_features(self, sequence):
        """
  Calculate the GTPC features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of GTPC feature values.
  """
        # Create a mapping of amino acids to their corresponding groups
        amino_acid_to_group = {}
        for group, amino_acids in self.amino_acid_groups.items():
            for amino_acid in amino_acids:
                amino_acid_to_group[amino_acid] = group

        # Generate all possible group tripeptide combinations
        group_tripeptides = [''.join(group) for group in product(self.group_names, repeat=3)]

        # Initialize a dictionary to store the counts of each group tripeptide
        group_tripeptide_counts = {group_tripeptide: 0 for group_tripeptide in group_tripeptides}

        # Count the occurrences of each group tripeptide in the sequence
        for i in range(len(sequence) - 2):
            tripeptide = sequence[i:i + 3]
            group_tripeptide = ''.join([amino_acid_to_group.get(amino_acid, '') for amino_acid in tripeptide])
            if group_tripeptide in group_tripeptide_counts:
                group_tripeptide_counts[group_tripeptide] += 1

        # Calculate the sequence length
        sequence_length = len(sequence)

        # Calculate the GTPC feature values using the formula
        gtpc_features = [group_tripeptide_counts[group_tripeptide] / (sequence_length - 2) for group_tripeptide in
                         group_tripeptides]

        return gtpc_features
