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
import shap
# import eli5
# import lime
# import optuna
# import hyperopt
# import scipy
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict# , GridSearchCV, KFold, RepeatedStratifiedKFold
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

pd.set_option('display.max_columns', 500)

TARGET_VAR = 'abandono_6meses'
N_FOLDS = 3
N_JOBS = os.cpu_count() // 2.5


# Steps:
# 1. Load data
deleted_columns = [# 'degree', 
                #    'ratio_salario_psi_score', 'ratio_salario_distancia', 
                #    'edad_cuando_se_incorporo', 'trimestre_incorporacion',
                #    'mes_incorporacion', 'edad',
                #    'performance_score'
                   ]
dt_train = pd.read_parquet('../data/processed/train_data.parquet').drop(columns=deleted_columns)
dt_test = pd.read_parquet('../data/processed/test_data.parquet').drop(columns=deleted_columns)

df_train = pd.merge(
    dt_train,
    dt_train.copy().drop(columns=['id_ultimo_jefe', TARGET_VAR]),#[['id_colaborador', 'edad', 'dias_baja_salud', 'performance_score', 'modalidad_trabajo', 'psi_score', 'ratio_salario_distancia', 'degree']],
    left_on='id_ultimo_jefe',
    right_on='id_colaborador',
    suffixes=('', '_jefe'),
    how='left'
)
df_test = pd.merge(
    dt_test,
    dt_test.copy().drop(columns=['id_ultimo_jefe']),#dt_test[['id_colaborador', 'edad', 'dias_baja_salud', 'performance_score', 'modalidad_trabajo', 'psi_score', 'ratio_salario_distancia', 'degree']],
    left_on='id_ultimo_jefe',
    right_on='id_colaborador',
    suffixes=('', '_jefe'),
    how='left'
)

categorical_cols = df_train.select_dtypes(include=['category']).columns.to_list()
numerical_cols = df_train.select_dtypes(include=['number']).columns.to_list()
other_cols = list(df_train.columns.difference(categorical_cols).difference(numerical_cols))
print(f'categorical_cols: {categorical_cols}')
print(f'numerical_cols: {numerical_cols}')
print(f'other_cols: {other_cols}')


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
train_cols = X.columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
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
            grow_policy='lossguide',
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

def create_stacking_models():
    base_models = list()
    base_models.append(('lgbm',  LGBMClassifier(
                            n_estimators=25,
                            min_gain_to_split=0.5,
                            max_depth=4,
                            learning_rate=0.05,
                            boosting_type='gbdt',
                            objective='binary',
                            random_state=SEED,
                            verbose=-1,
                        )
                       ))
    base_models.append(('xgb', XGBClassifier(
                                n_estimators=25,
                                max_depth=4,
                                objective='binary:logistic', 
                                tree_method='hist',
                                enable_categorical=True,
                                grow_policy='lossguide',
                                random_state=SEED,
                            )
                       ))    
    #base_models.append(('GNB', GaussianNB()))
    #base_models.append(('RF', RandomForestClassifier(n_estimators= 200, 
    #                                               oob_score = True, 
    #                                               class_weight = "balanced", 
    #                                               random_state = 20, 
    #                                               ccp_alpha = 0.1)
    #                   ))

    
    meta_model = LogisticRegression()
    final_model = StackingClassifier(estimators = base_models, ##Base estimators which will be stacked together
                                     final_estimator = meta_model,
                                     cv = 5
                                    )
    return final_model


model = create_stacking_models()
# get_model('xgb')
# get_model('lgbm')
# get_model('catboost', cat_features=categorical_cols)

FACTOR = 0.25
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

f1_scores = []
for X_train_idx, X_test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[X_train_idx], X.iloc[X_test_idx]
    y_train, y_test = y.iloc[X_train_idx], y.iloc[X_test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1] > FACTOR

    f1_scores.append(f1_score(y_test, y_pred, average='binary'))
print(f'f1_scores mean: {np.mean(f1_scores)}')
print(f'f1_scores std: {np.std(f1_scores)}')

# 5. Compare models
# 6. Tune hyperparameters
# 7. Make predictions
# 8. Save model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
model.fit(X_train, y_train)

# y_pred = model.predict(X_train)
y_pred = (model.predict_proba(X_train)[:, 1] > FACTOR).astype(int)
print(classification_report(y_train, y_pred))
# print(confusion_matrix(y_train, y_pred))

# y_pred = model.predict(X_test)
y_pred = (model.predict_proba(X_test)[:, 1] > FACTOR).astype(int)
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# 9. Load model
# 10. Interpret model
# 		- Feature importance (permutation, SHAP, etc.)
# 		- Partial dependence plots
# 		- Individual conditional expectation plots
# 		- LIME
# 		- ELI5
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame(sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True), columns=['feature', 'importance'])
feature_importances.plot(kind='barh', x='feature', y='importance', color='blue')

# shap.initjs()

# explainer = shap.TreeExplainer(model)

# shap_values = explainer.shap_values(X_test)
# print("Variable Importance Plot - Global Interpretation")
# figure = plt.figure()
# shap.summary_plot(shap_values, X_test)

# 11. Deploy model/Submit predictions
# 12. Monitor model
# print(X.columns)

df_test[TARGET_VAR] = (
    model.predict_proba(pd.get_dummies(df_test[train_cols], columns=categorical_cols, drop_first=True))
    [:, 1] > FACTOR
).astype(int)
print(df_test[TARGET_VAR].value_counts(normalize=True))
print(df_test[TARGET_VAR].value_counts())

best_submission = pd.read_csv('../submissions/xgb_all_join_jefe.csv')
print(best_submission[TARGET_VAR].value_counts(normalize=True))
best_submission['equal'] = (best_submission[TARGET_VAR] == df_test[TARGET_VAR])
print(best_submission['equal'].value_counts(normalize=True))

df_test[['id_colaborador', TARGET_VAR]].rename(columns={'id_colaborador': 'ID'})\
    .to_csv('../submissions/xgb_all_join_jefe_stacking.csv', index=False)



from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

X.fillna(X.mean(), inplace=True)

def create_stacking_models():
    base_models = list()
    base_models.append(('lgbm',  LGBMClassifier(
                            n_estimators=25,
                            min_gain_to_split=0.5,
                            max_depth=4,
                            learning_rate=0.05,
                            boosting_type='gbdt',
                            objective='binary',
                            random_state=SEED,
                            verbose=-1,
                        )
                       ))
    base_models.append(('xgb', XGBClassifier(
                                n_estimators=25,
                                max_depth=4,
                                objective='binary:logistic', 
                                tree_method='hist',
                                enable_categorical=True,
                                grow_policy='lossguide',
                                random_state=SEED,
                            )
                       ))    
    #base_models.append(('GNB', GaussianNB()))
    #base_models.append(('RF', RandomForestClassifier(n_estimators= 200, 
    #                                               oob_score = True, 
    #                                               class_weight = "balanced", 
    #                                               random_state = 20, 
    #                                               ccp_alpha = 0.1)
    #                   ))

    
    meta_model = LogisticRegression()
    final_model = StackingClassifier(estimators = base_models, ##Base estimators which will be stacked together
                                     final_estimator = meta_model,
                                     cv = 5
                                    )
    return final_model

def models_all():
    all_models = dict()
    all_models['lgbm']= LGBMClassifier(
                        n_estimators=25,
                        min_gain_to_split=0.5,
                        max_depth=4,
                        learning_rate=0.05,
                        boosting_type='gbdt',
                        objective='binary',
                        random_state=SEED,
                        verbose=-1,
                    )
    all_models['xgb']= XGBClassifier(
                                n_estimators=25,
                                max_depth=4,
                                objective='binary:logistic', 
                                tree_method='hist',
                                enable_categorical=True,
                                grow_policy='lossguide',
                                random_state=SEED,
                            )

    #all_models['RF']= RandomForestClassifier(n_estimators= 200, 
    #                                               oob_score = True, 
    #                                               class_weight = "balanced", 
    #                                               random_state = 20, 
    #                                               ccp_alpha = 0.15)
    #all_models['GNB'] = GaussianNB()
    all_models['Stacking'] = create_stacking_models()
    return all_models

def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='f1_weighted', cv=cv, error_score='raise')
    return scores


model_results = list()
models = models_all()
names = list()

# Create a for loop that iterates over each name, model in models dictionary 
for name, model in models.items():
    scores = evaluate_model(model)
    model_results.append(scores)
    names.append(name)
#     print(model_results)
    print('>%s %.3f (%.3f) \n' % (name, np.mean(scores), np.std(scores)))