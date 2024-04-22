from collections import defaultdict


class AAIFeatureExtraction:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aai_data = {
            'TSAJ990101': {'A': 0.48, 'R': 0.95, 'N': 0.27, 'D': 0.08, 'C': 1.38, 'Q': 0.22, 'E': 0.11, 'G': 0.00,
                           'H': 0.66, 'I': 2.22, 'L': 1.53, 'K': 1.15, 'M': 1.18, 'F': 2.12, 'P': 0.39, 'S': 0.19,
                           'T': 0.73, 'W': 2.66, 'Y': 1.61, 'V': 1.73},
            'LIFS790101': {'A': 0.52, 'R': -0.68, 'N': -0.70, 'D': -0.81, 'C': 0.25, 'Q': -0.41, 'E': -0.83, 'G': 0.00,
                           'H': -0.50, 'I': 2.46, 'L': 1.82, 'K': -0.63, 'M': 1.40, 'F': 2.44, 'P': -0.21, 'S': -0.36,
                           'T': -0.03, 'W': 2.26, 'Y': 1.39, 'V': 1.78},
            'MIYS990104': {'A': -0.02, 'R': -0.42, 'N': -0.77, 'D': -1.04, 'C': 0.77, 'Q': -0.91, 'E': -1.14,
                           'G': -0.80, 'H': 0.26, 'I': 1.81, 'L': 1.14, 'K': -0.41, 'M': 1.00, 'F': 1.35, 'P': -0.09,
                           'S': -0.97, 'T': -0.77, 'W': 1.71, 'Y': 1.11, 'V': 0.91},
            'CEDJ970104': {'A': 7.0, 'R': 93.0, 'N': 58.0, 'D': 40.0, 'C': 1.0, 'Q': 72.0, 'E': 83.0, 'G': 3.0,
                           'H': 83.0, 'I': 1.0, 'L': 1.0, 'K': 56.0, 'M': 10.0, 'F': 1.0, 'P': 55.0, 'S': 42.0,
                           'T': 32.0, 'W': 1.0, 'Y': 3.0, 'V': 3.0},
            'MAXF760101': {'A': 113.0, 'R': 241.0, 'N': 158.0, 'D': 151.0, 'C': 140.0, 'Q': 198.0, 'E': 183.0,
                           'G': 85.0, 'H': 202.0, 'I': 108.0, 'L': 137.0, 'K': 211.0, 'M': 160.0, 'F': 113.0, 'P': 57.0,
                           'S': 143.0, 'T': 146.0, 'W': 163.0, 'Y': 117.0, 'V': 105.0},
            'BIOV880101': {'A': 0.61, 'R': -0.39, 'N': -0.92, 'D': -1.31, 'C': 1.52, 'Q': -1.22, 'E': -1.61, 'G': 0.0,
                           'H': -0.64, 'I': 2.22, 'L': 1.53, 'K': -0.67, 'M': 1.18, 'F': 2.12, 'P': -0.49, 'S': -1.07,
                           'T': -1.21, 'W': 1.60, 'Y': 0.01, 'V': 1.73},
            'BLAM930101': {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497,
                           'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314,
                           'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386},
            'NAKH920108': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0,
                           'H': 10.4, 'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6,
                           'W': 5.4, 'Y': 6.2, 'V': 5.9}
        }

    def calculate_aai_features(self, sequence):
        """
  Calculate the AAI features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of AAI feature values.
  """
        # Initialize a dictionary to store the counts of each amino acid
        aa_counts = defaultdict(int)

        # Count the occurrences of each amino acid in the sequence
        for aa in sequence:
            if aa in self.amino_acids:
                aa_counts[aa] += 1

        # Calculate the total number of amino acids in the sequence
        total_aa = sum(aa_counts.values())

        # Initialize a list to store the AAI feature values
        aai_features = []

        # Calculate the AAI feature values for each amino acid index
        for index in self.aai_data:
            index_values = self.aai_data[index]
            for aa in self.amino_acids:
                if total_aa > 0:
                    aai_features.append(aa_counts[aa] * index_values[aa] / total_aa)
                else:
                    aai_features.append(0)

        return aai_features

# Example usage
# sequence = "ACDEFGHIKLMNPQRSTVWY"
# extractor = AAIFeatureExtraction()
# aai_features = extractor.calculate_aai_features(sequence)
# print(aai_features)
