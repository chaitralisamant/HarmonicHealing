import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load pretrained Random Forest model
model = joblib.load("stress_det.pkl")

# breakpoint()
# "MEAN_RR", "MEDIAN_RR", "SDRR", "SDRR_RMSSD", "HR", "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR"

def hr_to_features(hr):
    
    # estimating the values from heart rate alone
    
    mean_rr = 60000/hr
    median_rr = 60000/hr
    
    sdrr = 0.05 * (mean_rr)
    
    sdrr_rmssd = 1
    mean_rel_rr = 1
    median_rel_rr = 1
    sdrr_rel_rr = 0.05
    
    features = [[mean_rr, median_rr, sdrr, sdrr_rmssd, hr, mean_rel_rr, median_rel_rr, sdrr_rel_rr]]
    
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    standardized = scaler.transform(features)
    pca_features = pca.transform(standardized)
    
    return pca_features

input = hr_to_features(100)

# breakpoint()

prediction = model.predict(input)

print(prediction)