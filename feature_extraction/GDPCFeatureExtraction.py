from collections import Counter
from itertools import product


class GDPCFeatureExtraction:
    def __init__(self):
        self.amino_acid_groups = {
            'g1': 'FYW',  # Aromatic
            'g2': 'RKH',  # Positively charged
            'g3': 'GAVLMI',  # Aliphatic
            'g4': 'STCPNQ',  # Uncharged
            'g5': 'DE'  # Negatively charged
        }
        self.group_names = list(self.amino_acid_groups.keys())

    def calculate_gdpc_features(self, sequence):
        """
        Calculate the GDPC features for a given protein sequence.

        :param sequence: The protein sequence as a string.
        :return: A list of GDPC feature values.
        """
        # Create a mapping of amino acids to their corresponding groups
        amino_acid_to_group = {}
        for group, amino_acids in self.amino_acid_groups.items():
            for amino_acid in amino_acids:
                amino_acid_to_group[amino_acid] = group

        # Generate all possible group dipeptide combinations
        group_dipeptides = [''.join(group) for group in product(self.group_names, repeat=2)]

        # Initialize a dictionary to store the counts of each group dipeptide
        group_dipeptide_counts = {group_dipeptide: 0 for group_dipeptide in group_dipeptides}

        # Count the occurrences of each group dipeptide in the sequence
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            group_dipeptide = ''.join([amino_acid_to_group.get(amino_acid, '') for amino_acid in dipeptide])
            if group_dipeptide in group_dipeptide_counts:
                group_dipeptide_counts[group_dipeptide] += 1

        # Calculate the sequence length
        sequence_length = len(sequence)

        # Calculate the GDPC feature values using the formula
        gdpc_features = [group_dipeptide_counts[group_dipeptide] / (sequence_length - 1) for group_dipeptide in
                         group_dipeptides]

        return gdpc_features
