"""Answers to the self-check boxes of Chapter 4."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
Sxy, Sxx, a = 6, 4, 2
print('slope', Sxy / (Sxx + a))
print('columns for p=3, d=2 (powers only):', 3 * 2, ' with products:', PolynomialFeatures(2, include_bias=False).fit(np.zeros((1, 3))).n_output_features_)
