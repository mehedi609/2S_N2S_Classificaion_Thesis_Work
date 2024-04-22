from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut
import numpy as np

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

# Load the dataset
data = pd.read_excel('../data/Final_2Sm_modified_with_sequences.xlsx')

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder to the folding_type column and transform it to numeric labels
data['folding_type'] = label_encoder.fit_transform(data['folding_type'])

# Now, when you extract labels for model training:
labels = data['folding_type'].values

# Initialize the FeatureExtraction class
feature_extraction = KAACFeatureExtraction()

# Feature extraction using AAC with length
features = np.array([feature_extraction.calculate_kaac_features(seq) for seq in data['sequence']])

# SVM with Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
y_true, y_pred = [], []
for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred.append(clf.predict(X_test)[0])
    y_true.append(y_test[0])