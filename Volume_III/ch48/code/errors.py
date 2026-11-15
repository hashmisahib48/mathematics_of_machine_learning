# The error messages of Chapter 48.
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sensors import make_sensors

Z = make_sensors()[0]
trials = {
    "LOF(novelty=False).predict": lambda: LocalOutlierFactor().fit(Z).predict(Z),
    "LOF(novelty=True).fit_predict": lambda: LocalOutlierFactor(novelty=True).fit_predict(Z),
    "OneClassSVM(contamination=0.05)": lambda: OneClassSVM(contamination=0.05),
    "EllipticEnvelope(contamination='auto')": lambda: EllipticEnvelope(contamination="auto").fit(Z)}
for name, t in trials.items():
    try:
        t()
    except Exception as e:
        print(name, "->", type(e).__name__)
        print("   ", str(e).replace("\n", " ")[:150])
