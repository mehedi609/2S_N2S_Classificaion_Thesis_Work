import numpy as np


class QSOFeatureExtraction:
    def __init__(self, weight=0.1):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.weight = weight
        self.schneider_wrede_distance_matrix = np.array([
            [0.0, 0.94, 0.90, 0.82, 0.96, 0.65, 0.97, 0.58, 0.89, 1.36, 1.08, 0.69, 1.02, 0.54, 0.88, 0.81, 1.00, 1.37,
             0.56, 0.77],
            [0.94, 0.0, 0.22, 1.34, 0.16, 1.05, 0.16, 0.92, 0.28, 0.64, 0.50, 1.17, 0.26, 0.82, 0.30, 0.24, 0.43, 0.73,
             0.96, 0.41],
            [0.90, 0.22, 0.0, 1.24, 0.18, 1.03, 0.22, 0.83, 0.32, 0.81, 0.61, 1.07, 0.31, 0.71, 0.35, 0.13, 0.46, 0.90,
             0.87, 0.37],
            [0.82, 1.34, 1.24, 0.0, 1.41, 0.96, 1.43, 1.17, 1.17, 1.84, 1.59, 0.97, 1.45, 0.94, 1.19, 1.20, 1.39, 1.85,
             0.99, 1.26],
            [0.96, 0.16, 0.18, 1.41, 0.0, 1.08, 0.02, 0.94, 0.30, 0.54, 0.45, 1.20, 0.21, 0.85, 0.28, 0.28, 0.41, 0.63,
             0.99, 0.43],
            [0.65, 1.05, 1.03, 0.96, 1.08, 0.0, 1.12, 0.80, 0.93, 1.55, 1.25, 0.38, 1.10, 0.28, 0.95, 0.92, 0.97, 1.55,
             0.44, 0.87],
            [0.97, 0.16, 0.22, 1.43, 0.02, 1.12, 0.0, 0.95, 0.29, 0.57, 0.48, 1.23, 0.23, 0.89, 0.31, 0.29, 0.43, 0.66,
             1.02, 0.46],
            [0.58, 0.92, 0.83, 1.17, 0.94, 0.80, 0.95, 0.0, 0.84, 1.22, 0.95, 0.94, 0.89, 0.64, 0.81, 0.75, 0.85, 1.23,
             0.71, 0.75],
            [0.89, 0.28, 0.32, 1.17, 0.30, 0.93, 0.29, 0.84, 0.0, 0.71, 0.53, 1.08, 0.20, 0.76, 0.18, 0.28, 0.27, 0.80,
             0.90, 0.27],
            [1.36, 0.64, 0.81, 1.84, 0.54, 1.55, 0.57, 1.22, 0.71, 0.0, 0.55, 1.66, 0.65, 1.33, 0.73, 0.78, 0.80, 0.39,
             1.48, 0.83],
            [1.08, 0.50, 0.61, 1.59, 0.45, 1.25, 0.48, 0.95, 0.53, 0.55, 0.0, 1.34, 0.38, 1.01, 0.44, 0.57, 0.50, 0.52,
             1.14, 0.55],
            [0.69, 1.17, 1.07, 0.97, 1.20, 0.38, 1.23, 0.94, 1.08, 1.66, 1.34, 0.0, 1.17, 0.50, 1.06, 0.99, 1.13, 1.66,
             0.65, 1.03],
            [1.02, 0.26, 0.31, 1.45, 0.21, 1.10, 0.23, 0.89, 0.20, 0.65, 0.38, 1.17, 0.0, 0.88, 0.18, 0.31, 0.20, 0.73,
             1.00, 0.29],
            [0.54, 0.82, 0.71, 0.94, 0.85, 0.28, 0.89, 0.64, 0.76, 1.33, 1.01, 0.50, 0.88, 0.0, 0.76, 0.65, 0.81, 1.32,
             0.34, 0.65],
            [0.88, 0.30, 0.35, 1.19, 0.28, 0.95, 0.31, 0.81, 0.18, 0.73, 0.44, 1.06, 0.18, 0.76, 0.0, 0.31, 0.17, 0.81,
             0.87, 0.21],
            [0.81, 0.24, 0.13, 1.20, 0.28, 0.92, 0.29, 0.75, 0.28, 0.78, 0.57, 0.99, 0.31, 0.65, 0.31, 0.0, 0.40, 0.87,
             0.79, 0.33],
            [1.00, 0.43, 0.46, 1.39, 0.41, 0.97, 0.43, 0.85, 0.27, 0.80, 0.50, 1.13, 0.20, 0.81, 0.17, 0.40, 0.0, 0.88,
             0.99, 0.21],
            [1.37, 0.73, 0.90, 1.85, 0.63, 1.55, 0.66, 1.23, 0.80, 0.39, 0.52, 1.66, 0.73, 1.32, 0.81, 0.87, 0.88, 0.0,
             1.49, 0.90],
            [0.56, 0.96, 0.87, 0.99, 0.99, 0.44, 1.02, 0.71, 0.90, 1.48, 1.14, 0.65, 1.00, 0.34, 0.87, 0.79, 0.99, 1.49,
             0.0, 0.81],
            [0.77, 0.41, 0.37, 1.26, 0.43, 0.87, 0.46, 0.75, 0.27, 0.83, 0.55, 1.03, 0.29, 0.65, 0.21, 0.33, 0.21, 0.90,
             0.81, 0.0]
        ])

    def calculate_qso_features(self, sequence):
        """
  Calculate the QSO features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of QSO feature values.
  """
        sequence_length = len(sequence)
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        aa_frequencies = [aa_counts.get(aa, 0) / sequence_length for aa in self.amino_acids]

        tau_values = self._calculate_tau_values(sequence)

        denominator = sum(aa_frequencies) + self.weight * sum(tau_values)

        qso_features = []
        for i in range(20):
            qso_features.append(aa_frequencies[i] / denominator)
        for i in range(30):
            qso_features.append((self.weight * tau_values[i] - 20) / denominator)

        return qso_features

    def _calculate_tau_values(self, sequence):
        """
  Calculate the tau values for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of tau values.
  """
        tau_values = []
        sequence_length = len(sequence)

        for c in range(1, 31):
            tau_c = 0
            for i in range(sequence_length - c):
                aa1 = sequence[i]
                aa2 = sequence[i + c]
                if aa1 in self.amino_acids and aa2 in self.amino_acids:
                    dist = self.schneider_wrede_distance_matrix[self.amino_acids.index(aa1)][
                        self.amino_acids.index(aa2)]
                    tau_c += dist ** 2
            tau_values.append(tau_c)

        return tau_values

# Example usage
# sequence = "RKRQAWLWEEDKNLRSGVRKYGEGNWSKILLHYKFNNRTSVMLKDRWRTMKKL"
# extractor = QSOFeatureExtraction(weight=0.1)
# qso_features = extractor.calculate_qso_features(sequence)
# print(qso_features)
# print(len(qso_features))
# print(qso_features.shape)
