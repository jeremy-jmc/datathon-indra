
"""
https://www.kaggle.com/discussions/general/237792
https://notebook.community/minesh1291/MachineLearning/xgboost/feature_importance_v1
https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
https://medium.com/@emilykmarsh/xgboost-feature-importance-233ee27c33a4
https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
https://forecastegy.com/posts/xgboost-feature-importance-python/
https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-xgboost
"""

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

features = (
    pd.read_csv('./features.csv')
    .loc[lambda df: df['importance'] > 0.0]
)
display(features)

display(
    features.groupby('tipo_variable').agg({'importance': 'sum'}).sort_values('importance', ascending=False)
)

features['grupo'] = [
    'Puntaje evaluacion',
    'Dias de baja por salud',
    'Distancia a la oficina',
    'Salario',
    'Ratio salario-puntaje',
    'Puntaje test psicometrico',
    'Puntaje evaluacion del jefe',
    'Permanencia promedio',
    'Ratio salario-distancia',
    'Edad',
    'Edad cuando se incorporo',
    'Mes de incorporacion',
    'Interacciones',
    np.nan,
    'Puntaje test psicometrico del jefe',
    'Distancia a la oficina del jefe',
    'Canal de reclutamiento',
    'Estado civil',
    np.nan,
    'Edad del jefe',
    'Interacciones del jefe',
    'Estado civil del jefe',
    'Dias de baja por salud del jefe',
    'Ratio salario-distancia del jefe',
    'Modalidad de trabajo',
    'Canal de reclutamiento',
    np.nan,
    'Permanencia promedio del jefe',
    np.nan,
    np.nan,
    'Mes de incorporacion del jefe',
    np.nan,
    np.nan,
    'Canal de reclutamiento',
    'Trimestre de incorporacion',
    'Ratio salario-psicometrico del jefe',
    'Canal de reclutamiento del jefe',
    'Trimestre de incorporacion',
    'Modalidad de trabajo del jefe',
]

features_plot = (
    features
    .dropna(subset=['grupo'])
    .loc[lambda df: df['importance'] > 0.01]
    .reset_index(drop=True)
)

tipo_color = {'colaborador': '#d8315bff', 'jefe': '#0a2463ff'}

features_gb = (
    features_plot.groupby('grupo')
    .agg({'importance': 'sum', 'tipo_variable': 'first'})
    .reset_index()
    .sort_values('importance', ascending=True)
    .reset_index(drop=True)
)
features_gb['color'] = features_gb['tipo_variable'].map(tipo_color)
features_gb['tipo_variable'] = features_gb['tipo_variable'].astype('category')
features_gb = features_gb.iloc[-20:]

fig, ax = plt.subplots(figsize=(12, 20))
bars = ax.barh(features_gb['grupo'], 
    features_gb['importance'], 
    color=features_gb['color']
)

ax.set_xlabel('Importancia (%)')
ax.set_title('Feature Importance')

# Añadir etiquetas de porcentaje a las barras
for bar in bars:
    width = bar.get_width()
    label_x_pos = width if width > 0 else -1
    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
            f'  {width*100:.1f}%', va='center')
plt.legend()
plt.show()
