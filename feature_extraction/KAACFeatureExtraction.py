from collections import Counter


class KAACFeatureExtraction:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    def calculate_kaac_features(self, sequence):
        """
  Calculate the KAAC features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of KAAC feature values.
  """
        # Calculate the amino acid composition (AAC)
        aac_features = self.calculate_aac_features(sequence)

        # Calculate the sequence length (K)
        sequence_length = len(sequence)

        # Combine AAC features with sequence length
        kaac_features = aac_features + [sequence_length]

        return kaac_features

    def calculate_aac_features(self, sequence):
        """
  Calculate the amino acid composition (AAC) features.

  :param sequence: The protein sequence as a string.
  :return: A list of AAC feature values.
  """
        # Count the occurrences of each amino acid in the sequence
        amino_acid_counts = Counter(sequence)

        # Calculate the total number of amino acids in the sequence
        total_amino_acids = sum(amino_acid_counts.values())

        # Calculate the normalized frequency of each amino acid
        aac_features = [amino_acid_counts.get(aa, 0) / total_amino_acids for aa in self.amino_acids]

        return aac_features
