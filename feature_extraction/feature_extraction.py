from collections import Counter, defaultdict
from itertools import product
import numpy as np


class FeatureExtraction:
    # Assuming you will define AMINO_ACIDS and GROUPS here or pass them during initialization
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

    def __init__(self, amino_acids=None, groups=None, codon_counts=None, cn=None):
        if amino_acids is not None:
            self.amino_acids = amino_acids
        if groups is not None:
            self.groups = groups
        if codon_counts is not None:
            self.codon_counts = codon_counts
        if cn is not None:
            self.CN = cn

    def calculate_aac_with_length(self, sequence):
        sequence_length = len(sequence)
        aa_counts = Counter(sequence)
        aac_with_length = [aa_counts.get(aa, 0) / sequence_length for aa in self.amino_acids]
        aac_with_length.append(sequence_length)
        return aac_with_length

    def calculate_dpc(self, sequence):
        possible_dipeptides = [''.join(dp) for dp in product(self.amino_acids, repeat=2)]
        dipeptide_counts = Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))
        total_dipeptides = sum(dipeptide_counts.values())
        dpc = {dp: dipeptide_counts[dp] / total_dipeptides for dp in possible_dipeptides}
        return list(dpc.values())

    def calculate_gdpc(self, sequence):
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

            # Normalize by the number of possible pairs given the window size
            window_size = len(sequence) - k - 1
            features += [(k_spaced_pairs.get(pair, 0) / window_size if window_size > 0 else 0) for pair in possible_pairs]
        return features

    def calculate_ct(self, sequence):
        # Define the groups based on physicochemical properties
        groups = {
            1: 'AVG',
            2: 'TSYM',
            3: 'FLIP',
            4: 'HQNW',
            5: 'DE',
            6: 'RK',
            7: 'C'
        }

        # Reverse mapping from amino acid to group
        aa_to_group = {aa: group for group, aas in groups.items() for aa in aas}

        # Initialize a dictionary to count occurrences of each triad (represented by group numbers)
        triad_counts = {}

        # Form triads and convert them to group numbers
        for i in range(len(sequence) - 2):
            triad = ''.join([str(aa_to_group.get(aa, 0)) for aa in sequence[i:i+3]])
            triad_counts[triad] = triad_counts.get(triad, 0) + 1

        # Calculate frequencies
        total_triads = sum(triad_counts.values())
        # Create a 343-dimensional feature vector (7^3 for the groups)
        feature_vector = [triad_counts.get(f'{i}{j}{k}', 0) / total_triads for i in range(1, 8) for j in range(1, 8) for k in range(1, 8)]

        return feature_vector

    def calculate_tm_tv(self, x, y):
        # Get codon usage counts for amino acids x and y
        cx = self.codon_counts.get(x, 0)
        cy = self.codon_counts.get(y, 0)
        # Calculate the theoretical mean Tm(x, y)
        Tm = (cx / self.CN) * (cy / self.CN)
        # Calculate the theoretical variance Tv(x, y)
        Tv = Tm * (1 - Tm) / (self.CN - 1)  # Using CN-1 since stop codons are excluded
        return Tm, Tv

    def calculate_dde(self, sequence):
        dpc_values = self.calculate_dpc(sequence)
        dde_values = []
        for i, dp in enumerate(product(self.amino_acids, repeat=2)):
            x, y = dp
            Tm, Tv = self.calculate_tm_tv(x, y)
            # Ensure Tv is not zero to avoid division by zero error
            dde = (dpc_values[i] - Tm) / np.sqrt(Tv) if Tv > 0 else 0
            dde_values.append(dde)
        return dde_values

#%%
