# # ALGORITHMS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
# import mlflow
# from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
# from imblearn.ensemble import BalancedBaggingClassifier
# from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
# from lightgbm import LGBMClassifier, LGBMRegressor
# from xgboost import XGBClassifier, XGBRegressor
# import pycaret
# import shap
# import eli5
# import lime
# import optuna
# import hyperopt
# import scipy
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score
# from sklearn.metrics import f1_score, confusion_matrix, classification_report, jaccard_score
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import KDTree
# from sklearn.clustering import KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.impute import KNNImputer	# MissRanger
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
# from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor, RandomForestClassifier
# from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Steps:
# 1. Load data
# 2. Prepare data
# 	Feature engineering:
# 		- Squares, logs, cubics, etc. for skewed data
# 		- Ratios, variable interactions, daily variations, moving averages, EWMA, EMA, smoothing
# 		- Binning (percentile, decile), frequency encoding, embedding, etc. for categorical data
# 		- Outlier detection and removal
# 		- Missing value imputation (mean, median, mode, KNN, MICE, etc.)
# 		- Categorize the numerical data (binning, etc.)

# 		- Feature selection (correlation, multicollinearity, PCA, etc.)
# 		- Feature scaling (standardization, normalization, etc.)
# 		- Feature encoding (one-hot, label, ordinal, etc.)
# 		- Feature generation (polynomials, etc.)
# 		- Feature extraction (PCA, LDA, etc.)
# 		- Feature transformation (log, square, etc.)
# 		- Oversampling/undersampling (SMOTE, etc.)
# 3. Build a baseline model
#  Classification
# 		- Check class imbalance (class weights argument)
# 		- Stratified K-fold cross-validation
# 		- Threshold moving/tuning (punto de corte)
# 		- Search optimal threshold in a grid
# 		- Check classification report
# 		- Check confusion matrix
#  Regression
# 		- Check distribution of target variable
# 		- Check skewness of target variable
# 		- Check outliers
# 		- Check correlation with other variables
# 		- Check distribution of features
# 		- Check skewness of features
# 		- Check Kaggle competition notebooks
# 4. Build a model
# 5. Compare models
# 6. Tune hyperparameters
# 7. Make predictions
# 8. Save model
# 9. Load model
# 10. Interpret model
# 		- Feature importance (permutation, SHAP, etc.)
# 		- Partial dependence plots
# 		- Individual conditional expectation plots
# 		- LIME
# 		- ELI5
# 11. Deploy model/Submit predictions
# 12. Monitor model
