from collections import defaultdict


class CTFeatureExtraction:
    def __init__(self):
        self.amino_acid_groups = {
            1: 'AVG',
            2: 'TSYM',
            3: 'FLIP',
            4: 'HQNW',
            5: 'DE',
            6: 'RK',
            7: 'C'
        }

    def calculate_ct_features(self, sequence):
        """
  Calculate the CT features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of CT feature values.
  """
        # Create a mapping of amino acids to their corresponding groups
        aa_to_group = {}
        for group, amino_acids in self.amino_acid_groups.items():
            for aa in amino_acids:
                aa_to_group[aa] = group

        # Initialize a dictionary to store the counts of each triad
        triad_counts = defaultdict(int)

        # Iterate over the sequence and count the occurrences of each triad
        for i in range(len(sequence) - 2):
            triad = ''.join([str(aa_to_group.get(aa, 0)) for aa in sequence[i:i + 3]])
            triad_counts[triad] += 1

        # Calculate the total number of triads
        total_triads = sum(triad_counts.values())

        # Initialize a list to store the CT feature values
        ct_features = []

        # Calculate the CT feature values for each possible triad
        for i in range(1, len(self.amino_acid_groups) + 1):
            for j in range(1, len(self.amino_acid_groups) + 1):
                for k in range(1, len(self.amino_acid_groups) + 1):
                    triad = f'{i}{j}{k}'
                    if total_triads > 0:
                        ct_features.append(triad_counts[triad] / total_triads)
                    else:
                        ct_features.append(0)

        return ct_features

# Example usage
# sequence = "ACDEFGHIKLMNPQRSTVWY"
# extractor = CTFeatureExtraction()
# ct_features = extractor.calculate_ct_features(sequence)
# print(ct_features)
