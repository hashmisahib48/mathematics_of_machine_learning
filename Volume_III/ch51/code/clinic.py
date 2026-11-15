"""The six-patient clinic table of Chapter 51 (invented)."""
import numpy as np
nan = np.nan
COLS = ["age", "bp", "glucose", "bmi"]
X = np.array([[50, 130, 100, 28],
              [60, nan, 110, 30],
              [40, 120, nan, 24],
              [nan, 140, 120, 32],
              [55, 135, 105, nan],
              [45, 125, 95, 26]])
