# # ALGORITHMS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import os
# import mlflow
# from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
# from imblearn.ensemble import BalancedBaggingClassifier
from catboost import CatBoostClassifier#, CatBoostRegressor, Pool, cv
from lightgbm import LGBMClassifier#, LGBMRegressor
from xgboost import XGBClassifier#, XGBRegressor
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

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split# , GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report# , jaccard_score
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree, KNeighborsClassifier
# from sklearn.clustering import KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.impute import KNNImputer	# MissRanger
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, ExtraTreeClassifier
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
# from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Steps:
# 1. Load data
deleted_columns = ['degree', 
                #    'ratio_salario_psi_score', 'ratio_salario_distancia', 
                #    'edad_cuando_se_incorporo', 'trimestre_incorporacion',
                #    'mes_incorporacion', 'edad',
                #    'performance_score'
                   ]
df_train = pd.read_parquet('../data/processed/train_data.parquet').drop(columns=deleted_columns)
df_test = pd.read_parquet('../data/processed/test_data.parquet').drop(columns=deleted_columns)

categorical_cols = df_train.select_dtypes(include=['category']).columns.to_list()
numerical_cols = df_train.select_dtypes(include=['number']).columns.to_list()
other_cols = list(df_train.columns.difference(categorical_cols).difference(numerical_cols))
print(f'categorical_cols: {categorical_cols}')
print(f'numerical_cols: {numerical_cols}')
print(f'other_cols: {other_cols}')

TARGET_VAR = 'abandono_6meses'
N_FOLDS = 10
N_JOBS = os.cpu_count() // 2.5

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

data_train = df_train[categorical_cols + numerical_cols]
print(data_train.columns)
print(data_train[TARGET_VAR].value_counts(normalize=True))

X = data_train.drop(columns=[TARGET_VAR])
X = pd.get_dummies(X)
y = data_train[TARGET_VAR]

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

def get_model(model_name, **kwargs):
    if model_name == 'catboost':
        return CatBoostClassifier(
            iterations=25,
            depth=4,
            loss_function='Logloss',
            eval_metric='F1',
            random_state=SEED,
            verbose=0,
            cat_features=kwargs.get('cat_features', None),
            # ! GPU training is non-deterministic https://catboost.ai/en/docs/features/training-on-gpu
            # task_type='GPU',
            # devices='0:1',
        )
    elif model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=25,
            max_depth=4,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
    elif model_name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=25,
            max_depth=4,
            random_state=SEED,
        )
    elif model_name == 'lgbm':
        # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        return LGBMClassifier(
            n_estimators=25,
            min_gain_to_split=0.5,
            max_depth=4,
            learning_rate=0.05,
            boosting_type='gbdt',
            objective='binary',
            random_state=SEED,
            verbose=-1,
        )
    elif model_name == 'xgb':
        return XGBClassifier(
            n_estimators=25,
            max_depth=4,
            objective='binary:logistic', 
            tree_method='hist',
            enable_categorical=True,
            random_state=SEED,
        )
    
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# f1_binary = []
# for train_idx, test_idx in skf.split(X, y):
#     model = get_model('xgb')
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     f1_binary.append(f1_score(y_test, y_pred, average='binary'))
# print(f'f1_binary: {np.mean(f1_binary)}')
# print(f'f1_binary: {np.std(f1_binary)}')

model = OneVsOneClassifier(LinearSVC(dual="auto", random_state=42))
# SVC(random_state=SEED, probability=True)
# get_model('xgb')
# get_model('lgbm')
# get_model('catboost', cat_features=categorical_cols)

scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# 5. Compare models
# 6. Tune hyperparameters
# 7. Make predictions
# 8. Save model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
model.fit(X_train, y_train)

FACTOR = 0.48
y_pred = model.predict(X_train)
# y_pred = (model.predict_proba(X_train)[:, 1] > FACTOR).astype(int)
print(classification_report(y_train, y_pred))
# print(confusion_matrix(y_train, y_pred))

y_pred = model.predict(X_test)
# y_pred = (model.predict_proba(X_test)[:, 1] > FACTOR).astype(int)
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# 9. Load model
# 10. Interpret model
# 		- Feature importance (permutation, SHAP, etc.)
# 		- Partial dependence plots
# 		- Individual conditional expectation plots
# 		- LIME
# 		- ELI5
# importances = model.feature_importances_
# feature_names = X.columns
# feature_importances = pd.DataFrame(sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True), columns=['feature', 'importance'])
# feature_importances.plot(kind='barh', x='feature', y='importance', color='blue', figsize=(10, 6))


# 11. Deploy model/Submit predictions
# 12. Monitor model
print(X.columns)
df_test = pd.get_dummies(df_test)
df_test[TARGET_VAR] = model.predict(df_test[X.columns])
print(df_test[TARGET_VAR].value_counts(normalize=True))
print(df_test[TARGET_VAR].value_counts())

# df_test[['id_colaborador', TARGET_VAR]].rename(columns={'id_colaborador': 'ID'})\
#     .to_csv('../submissions/submission.csv', index=False)