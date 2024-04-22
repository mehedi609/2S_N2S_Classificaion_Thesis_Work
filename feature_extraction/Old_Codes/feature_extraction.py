from collections import Counter, defaultdict
from itertools import product
import numpy as np


class FeatureExtraction:
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    groups = {
        'g1': 'FYW',
        'g2': 'RKH',
        'g3': 'GAVLMI',
        'g4': 'STCPNQ',
        'g5': 'DE'
    }

    codon_counts = {
        'A': 4, 'R': 6, 'N': 2, 'D': 2, 'C': 2,
        'Q': 2, 'E': 2, 'G': 4, 'H': 2, 'I': 3,
        'L': 6, 'K': 2, 'M': 1, 'F': 2, 'P': 4,
        'S': 6, 'T': 4, 'W': 1, 'Y': 2, 'V': 4,
    }
    CN = 61  # Total number of possible codons excluding three stop codons

    # aai_data = {
    #     'TSAJ990101': {'A': 0.48, 'R': 0.95, 'N': 0.27, 'D': 0.08, 'C': 1.38, 'Q': 0.22, 'E': 0.11, 'G': 0.00, 'H': 0.66, 'I': 2.22, 'L': 1.53, 'K': 1.15, 'M': 1.18, 'F': 2.12, 'P': 0.39, 'S': 0.19, 'T': 0.73, 'W': 2.66, 'Y': 1.61, 'V': 1.73},
    #     'LIFS790101': {'A': 0.52, 'R': -0.68, 'N': -0.70, 'D': -0.81, 'C': 0.25, 'Q': -0.41, 'E': -0.83, 'G': 0.00, 'H': -0.50, 'I': 2.46, 'L': 1.82, 'K': -0.63, 'M': 1.40, 'F': 2.44, 'P': -0.21, 'S': -0.36, 'T': -0.03, 'W': 2.26, 'Y': 1.39, 'V': 1.78},
    #     'MIYS990104': {'A': -0.02, 'R': -0.42, 'N': -0.77, 'D': -1.04, 'C': 0.77, 'Q': -0.91, 'E': -1.14, 'G': -0.80, 'H': 0.26, 'I': 1.81, 'L': 1.14, 'K': -0.41, 'M': 1.00, 'F': 1.35, 'P': -0.09, 'S': -0.97, 'T': -0.77, 'W': 1.71, 'Y': 1.11, 'V': 0.91},
    #     'CEDJ970104': {'A': 7.0, 'R': 93.0, 'N': 58.0, 'D': 40.0, 'C': 1.0, 'Q': 72.0, 'E': 83.0, 'G': 3.0, 'H': 83.0, 'I': 1.0, 'L': 1.0, 'K': 56.0, 'M': 10.0, 'F': 1.0, 'P': 55.0, 'S': 42.0, 'T': 32.0, 'W': 1.0, 'Y': 3.0, 'V': 3.0},
    #     'MAXF760101': {'A': 113.0, 'R': 241.0, 'N': 158.0, 'D': 151.0, 'C': 140.0, 'Q': 198.0, 'E': 183.0, 'G': 85.0, 'H': 202.0, 'I': 108.0, 'L': 137.0, 'K': 211.0, 'M': 160.0, 'F': 113.0, 'P': 57.0, 'S': 143.0, 'T': 146.0, 'W': 163.0, 'Y': 117.0, 'V': 105.0},
    #     'BIOV880101': {'A': 0.61, 'R': -0.39, 'N': -0.92, 'D': -1.31, 'C': 1.52, 'Q': -1.22, 'E': -1.61, 'G': 0.0, 'H': -0.64, 'I': 2.22, 'L': 1.53, 'K': -0.67, 'M': 1.18, 'F': 2.12, 'P': -0.49, 'S': -1.07, 'T': -1.21, 'W': 1.60, 'Y': 0.01, 'V': 1.73},
    #     'BLAM930101': {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386},
    #     'NAKH920108': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9}
    # }
    #
    # property_groups = {
    #     'hydrophobicity': {'P': 'RKEDQN', 'H': 'GASTPHY', 'N': 'CLVIMFW'},
    #     'normalized_vdw': {'P': 'GASTPD', 'H': 'NVEQIL', 'N': 'MHKFRYW'},
    #     'polarity': {'P': 'LIFWCMVY', 'H': 'PATGS', 'N': 'HQRKNED'},
    #     'polarizability': {'P': 'GASDT', 'H': 'CPNVEQIL', 'N': 'KMHFRYW'},
    #     'charge': {'P': 'KR', 'H': 'ANCQGHILMFPSTWYV', 'N': 'DE'},
    #     'solvent_accessibility': {'P': 'ALFCGIVW', 'H': 'RKQEND', 'N': 'MPSTHY'},
    #     'secondary_structure': {'P': 'EALMQKRH', 'H': 'VIYCWFT', 'N': 'GNPSD'}
    # }

    def __init__(self, amino_acids=None, groups=None, codon_counts=None, cn=None):
        if amino_acids is not None:
            self.amino_acids = amino_acids
        if groups is not None:
            self.groups = groups
        if codon_counts is not None:
            self.codon_counts = codon_counts
        if cn is not None:
            self.CN = cn

        self.property_groups = {
            'hydrophobicity': {'P': 'RKEDQN', 'H': 'GASTPHY', 'N': 'CLVIMFW'},
            'normalized_vdw': {'P': 'GASTPD', 'H': 'NVEQIL', 'N': 'MHKFRYW'},
            'polarity': {'P': 'LIFWCMVY', 'H': 'PATGS', 'N': 'HQRKNED'},
            'polarizability': {'P': 'GASDT', 'H': 'CPNVEQIL', 'N': 'KMHFRYW'},
            'charge': {'P': 'KR', 'H': 'ANCQGHILMFPSTWYV', 'N': 'DE'},
            'solvent_accessibility': {'P': 'ALFCGIVW', 'H': 'RKQEND', 'N': 'MPSTHY'},
            'secondary_structure': {'P': 'EALMQKRH', 'H': 'VIYCWFT', 'N': 'GNPSD'}
        }

        self.aai_data = {
            'TSAJ990101': {'A': 0.48, 'R': 0.95, 'N': 0.27, 'D': 0.08, 'C': 1.38, 'Q': 0.22, 'E': 0.11, 'G': 0.00, 'H': 0.66, 'I': 2.22, 'L': 1.53, 'K': 1.15, 'M': 1.18, 'F': 2.12, 'P': 0.39, 'S': 0.19, 'T': 0.73, 'W': 2.66, 'Y': 1.61, 'V': 1.73},
            'LIFS790101': {'A': 0.52, 'R': -0.68, 'N': -0.70, 'D': -0.81, 'C': 0.25, 'Q': -0.41, 'E': -0.83, 'G': 0.00, 'H': -0.50, 'I': 2.46, 'L': 1.82, 'K': -0.63, 'M': 1.40, 'F': 2.44, 'P': -0.21, 'S': -0.36, 'T': -0.03, 'W': 2.26, 'Y': 1.39, 'V': 1.78},
            'MIYS990104': {'A': -0.02, 'R': -0.42, 'N': -0.77, 'D': -1.04, 'C': 0.77, 'Q': -0.91, 'E': -1.14, 'G': -0.80, 'H': 0.26, 'I': 1.81, 'L': 1.14, 'K': -0.41, 'M': 1.00, 'F': 1.35, 'P': -0.09, 'S': -0.97, 'T': -0.77, 'W': 1.71, 'Y': 1.11, 'V': 0.91},
            'CEDJ970104': {'A': 7.0, 'R': 93.0, 'N': 58.0, 'D': 40.0, 'C': 1.0, 'Q': 72.0, 'E': 83.0, 'G': 3.0, 'H': 83.0, 'I': 1.0, 'L': 1.0, 'K': 56.0, 'M': 10.0, 'F': 1.0, 'P': 55.0, 'S': 42.0, 'T': 32.0, 'W': 1.0, 'Y': 3.0, 'V': 3.0},
            'MAXF760101': {'A': 113.0, 'R': 241.0, 'N': 158.0, 'D': 151.0, 'C': 140.0, 'Q': 198.0, 'E': 183.0, 'G': 85.0, 'H': 202.0, 'I': 108.0, 'L': 137.0, 'K': 211.0, 'M': 160.0, 'F': 113.0, 'P': 57.0, 'S': 143.0, 'T': 146.0, 'W': 163.0, 'Y': 117.0, 'V': 105.0},
            'BIOV880101': {'A': 0.61, 'R': -0.39, 'N': -0.92, 'D': -1.31, 'C': 1.52, 'Q': -1.22, 'E': -1.61, 'G': 0.0, 'H': -0.64, 'I': 2.22, 'L': 1.53, 'K': -0.67, 'M': 1.18, 'F': 2.12, 'P': -0.49, 'S': -1.07, 'T': -1.21, 'W': 1.60, 'Y': 0.01, 'V': 1.73},
            'BLAM930101': {'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346, 'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462, 'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509, 'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386},
            'NAKH920108': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9}
        }

        self.codons = {
            'A': 4, 'R': 6, 'N': 2, 'D': 2, 'C': 2,
            'Q': 2, 'E': 2, 'G': 4, 'H': 2, 'I': 3,
            'L': 6, 'K': 2, 'M': 1, 'F': 2, 'P': 4,
            'S': 6, 'T': 4, 'W': 1, 'Y': 2, 'V': 4
        }
        self.total_codons = sum(self.codons.values())

    def calculate_aac_with_length(self, sequence):
        """
        Calculate the amino acid composition (AAC) with sequence length.

        :param sequence: The amino acid sequence.
        :return: A list of AAC values with sequence length appended.
        """
        sequence_length = len(sequence)
        aa_counts = Counter(sequence)
        aac_with_length = [aa_counts.get(aa, 0) / sequence_length for aa in self.amino_acids]
        aac_with_length.append(sequence_length)
        return aac_with_length

    def calculate_dpc(self, sequence):
        """
        Calculate the dipeptide composition (DPC) of the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of DPC values.
        """
        possible_dipeptides = [''.join(dp) for dp in product(self.amino_acids, repeat=2)]
        dipeptide_counts = Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))
        total_dipeptides = sum(dipeptide_counts.values())
        dpc = [dipeptide_counts[dp] / total_dipeptides for dp in possible_dipeptides]
        return dpc

    def calculate_gdpc(self, sequence):
        """
        Calculate the grouped dipeptide composition (GDPC) of the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of GDPC values.
        """
        group_mapping = {aa: group for group, aas in self.groups.items() for aa in aas}
        group_pairs = [''.join(gp) for gp in product(self.groups.keys(), repeat=2)]
        dipeptide_counts = defaultdict(int)
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            group_pair = ''.join(group_mapping.get(aa, '') for aa in dipeptide)
            if group_pair in group_pairs:
                dipeptide_counts[group_pair] += 1
        total = sum(dipeptide_counts.values())
        gdpc = [dipeptide_counts[gp] / total if total > 0 else 0 for gp in group_pairs]
        return gdpc

    def calculate_gtpc(self, sequence):
        """
        Calculate the grouped tripeptide composition (GTPC) of the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of GTPC values.
        """
        group_mapping = {aa: group for group, aas in self.groups.items() for aa in aas}
        group_triplets = [''.join(gt) for gt in product(self.groups.keys(), repeat=3)]
        tripeptide_counts = defaultdict(int)
        for i in range(len(sequence) - 2):
            tripeptide = sequence[i:i + 3]
            group_triplet = ''.join(group_mapping.get(aa, '') for aa in tripeptide)
            if group_triplet in group_triplets:
                tripeptide_counts[group_triplet] += 1
        total = sum(tripeptide_counts.values())
        gtpc = [tripeptide_counts[gt] / total if total > 0 else 0 for gt in group_triplets]
        return gtpc

    def calculate_cksaap(self, sequence, kmax=5):
        """
        Calculate the composition of k-spaced amino acid pairs (CKSAAP) of the sequence.

        :param sequence: The amino acid sequence.
        :param kmax: The maximum spacing between amino acid pairs (default: 5).
        :return: A list of CKSAAP values.
        """
        possible_pairs = [''.join(pair) for pair in product(self.amino_acids, repeat=2)]
        features = []

        for k in range(kmax + 1):
            k_spaced_pairs = {}
            for i in range(len(sequence) - k - 1):
                pair = sequence[i] + sequence[i + k + 1]
                if pair in possible_pairs:
                    if pair not in k_spaced_pairs:
                        k_spaced_pairs[pair] = 1
                    else:
                        k_spaced_pairs[pair] += 1

            window_size = len(sequence) - k - 1
            features += [(k_spaced_pairs.get(pair, 0) / window_size if window_size > 0 else 0) for pair in possible_pairs]
        return features

    def calculate_ct(self, sequence):
        """
        Calculate the composition of triads (CT) of the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of CT values.
        """
        groups = {
            1: 'AVG',
            2: 'TSYM',
            3: 'FLIP',
            4: 'HQNW',
            5: 'DE',
            6: 'RK',
            7: 'C'
        }

        aa_to_group = {aa: group for group, aas in groups.items() for aa in aas}
        triad_counts = {}

        for i in range(len(sequence) - 2):
            triad = ''.join([str(aa_to_group.get(aa, 0)) for aa in sequence[i:i+3]])
            triad_counts[triad] = triad_counts.get(triad, 0) + 1

        total_triads = sum(triad_counts.values())
        feature_vector = [triad_counts.get(f'{i}{j}{k}', 0) / total_triads for i in range(1, len(groups) + 1) for j in range(1, len(groups) + 1) for k in range(1, len(groups) + 1)]

        return feature_vector

    def calculate_tm_tv(self, x, y):
        """
        Calculate the theoretical mean (Tm) and theoretical variance (Tv) for amino acids x and y.

        :param x: The first amino acid.
        :param y: The second amino acid.
        :return: A tuple containing Tm and Tv values.
        """
        cx = self.codon_counts.get(x, 0)
        cy = self.codon_counts.get(y, 0)
        Tm = (cx / self.CN) * (cy / self.CN)
        Tv = Tm * (1 - Tm) / (self.CN - 1)  # Using CN-1 since stop codons are excluded
        return Tm, Tv

    def calculate_dde(self, sequence):
        """
        Calculate the deviation from expected dipeptide composition (DDE) of the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of DDE values.
        """
        dpc_values = self.calculate_dpc(sequence)
        dde_values = []
        for i, dp in enumerate(product(self.amino_acids, repeat=2)):
            x, y = dp
            Tm, Tv = self.calculate_tm_tv(x, y)
            dde = (dpc_values[i] - Tm) / np.sqrt(Tv) if Tv > 0 else 0
            dde_values.append(dde)
        return dde_values

    def calculate_aai(self, sequence):
        """
        Extract amino acid index (AAI) features from the sequence.

        :param sequence: The amino acid sequence.
        :return: A list of AAI feature values.
        """
        sequence = sequence.upper()
        feature_vector = []

        for index in self.aai_data:
            aa_counts = {aa: 0 for aa in self.aai_data[index]}
            for aa in sequence:
                if aa in aa_counts:
                    aa_counts[aa] += 1

            total_count = sum(aa_counts.values())
            normalized_counts = [count / total_count for count in aa_counts.values()]
            feature_vector.extend(normalized_counts)

        return feature_vector

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
                elif sequence[i] in property_groups[group_pair[1]] and sequence[i + 1] in property_groups[group_pair[0]]:
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