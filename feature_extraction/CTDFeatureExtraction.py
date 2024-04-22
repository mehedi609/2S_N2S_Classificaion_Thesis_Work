from collections import defaultdict


class CTDFeatureExtraction:
    def __init__(self):
        self.property_groups = {
            'hydrophobicity': {'P': 'RKEDQN', 'H': 'GASTPHY', 'N': 'CLVIMFW'},
            'normalized_vdw': {'P': 'GASTPD', 'H': 'NVEQIL', 'N': 'MHKFRYW'},
            'polarity': {'P': 'LIFWCMVY', 'H': 'PATGS', 'N': 'HQRKNED'},
            'polarizability': {'P': 'GASDT', 'H': 'CPNVEQIL', 'N': 'KMHFRYW'},
            'charge': {'P': 'KR', 'H': 'ANCQGHILMFPSTWYV', 'N': 'DE'},
            'solvent_accessibility': {'P': 'ALFCGIVW', 'H': 'RKQEND', 'N': 'MPSTHY'},
            'secondary_structure': {'P': 'EALMQKRH', 'H': 'VIYCWFT', 'N': 'GNPSD'}
        }

    def calculate_ctd_features(self, sequence):
        """
  Calculate the CTD features for a given protein sequence.

  :param sequence: The protein sequence as a string.
  :return: A list of CTD feature values.
  """
        ctd_features = []

        for property_name, property_groups in self.property_groups.items():
            c_features = self._calculate_c_features(sequence, property_groups)
            t_features = self._calculate_t_features(sequence, property_groups)
            d_features = self._calculate_d_features(sequence, property_groups)
            ctd_features.extend(c_features + t_features + d_features)

        return ctd_features

    def _calculate_c_features(self, sequence, property_groups):
        """
  Calculate the composition (C) features.

  :param sequence: The protein sequence as a string.
  :param property_groups: The property groups dictionary.
  :return: A list of C feature values.
  """
        c_features = []
        sequence_length = len(sequence)

        for group in ['P', 'H', 'N']:
            count = sum(1 for aa in sequence if aa in property_groups[group])
            c_features.append(count / sequence_length)

        return c_features

    def _calculate_t_features(self, sequence, property_groups):
        """
  Calculate the transition (T) features.

  :param sequence: The protein sequence as a string.
  :param property_groups: The property groups dictionary.
  :return: A list of T feature values.
  """
        t_features = []
        sequence_length = len(sequence)

        for group_pair in [('N', 'P'), ('H', 'N'), ('P', 'H')]:
            count = 0
            for i in range(sequence_length - 1):
                if sequence[i] in property_groups[group_pair[0]] and sequence[i + 1] in property_groups[group_pair[1]]:
                    count += 1
                elif sequence[i] in property_groups[group_pair[1]] and sequence[i + 1] in property_groups[
                    group_pair[0]]:
                    count += 1
            t_features.append(count / (sequence_length - 1))

        return t_features

    def _calculate_d_features(self, sequence, property_groups):
        """
  Calculate the distribution (D) features.

  :param sequence: The protein sequence as a string.
  :param property_groups: The property groups dictionary.
  :return: A list of D feature values.
  """
        d_features = []
        sequence_length = len(sequence)

        for group in ['P', 'H', 'N']:
            indices = [i for i, aa in enumerate(sequence) if aa in property_groups[group]]
            if indices:
                d_features.append(indices[0] / sequence_length)
                d_features.append((indices[-1] - indices[0] + 1) / sequence_length)
                d_features.append((indices[-1] + 1) / sequence_length)
                d_features.append(len(indices) / sequence_length)
                d_features.append(sum(indices) / (sequence_length * len(indices)))
            else:
                d_features.extend([0] * 5)

        return d_features

# Example usage
# sequence = "ACDEFGHIKLMNPQRSTVWY"
# extractor = CTDFeatureExtraction()
# ctd_features = extractor.calculate_ctd_features(sequence)
# print(ctd_features)
