import os
import sys
import re
import time
import datetime
import random
import warnings
import itertools as it
from collections import Counter
import math
import json
import pickle
import requests
import shutil
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
# import geopandas as gpd
# from pandasql import sqldf
from natsort import natsorted
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from tqdm import tqdm
from functools import reduce
from PIL import Image
import uuid
import gc
import gzip
from cycler import cycler
# import networkx as nx
# import osmnx as ox
# import plotly.graph_objects as go
# import folium
# import textdistance as td
# from colorama import init, Fore, Back, Style
# from missingpy import MissForest
# import missingno as msno

sns.set_style('whitegrid')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 3)
pd.options.display.float_format = lambda x : '{:,.2f}'.format(x) if (np.isnan(x) | np.isinf(x)) else '{:,.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)

colors = cycler(color=plt.get_cmap('tab10').colors)  # ['b', 'r', 'g']
mpl.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = 'lightgray'
mpl.rcParams['axes.prop_cycle'] = colors
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['figure.dpi'] = 100

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

ds_type = 'train'
df = pd.read_csv(f'../data/raw/{ds_type}_data.csv', sep=';')

df['id_colaborador'] = df['id_colaborador'].astype('string')
df['id_ultimo_jefe'] = df['id_ultimo_jefe'].astype('object')
df['seniority'] = df['seniority'].replace({1: 'No', 2: 'Si'}).astype('category')
df['modalidad_trabajo'] = df['modalidad_trabajo'].astype('category')
# distancia_oficina
# dias_baja_salud
df['genero'] = df['genero'].astype('category')
df['canal_reclutamiento'] = df['canal_reclutamiento'].astype('category')
# permanencia_promedio
df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], format='%d/%m/%Y')
df['edad'] = (datetime.datetime.now() - df['fecha_nacimiento']).dt.days // 365
# salario
# performance_score
# psi_score
df['fecha_incorporacion'] = pd.to_datetime(df['fecha_incorporacion'], format='%d/%m/%Y')
df['mes_incorporacion'] = df['fecha_incorporacion'].dt.month
df['tiempo_empresa'] = (datetime.datetime.now() - df['fecha_incorporacion']).dt.days // 365
df['estado_civil'] = df['estado_civil'].astype('category')
# df['abandono_6meses'] = df['abandono_6meses'].astype('category')


df = df.rename(columns={
    'seniority': 'personal_a_cargo',
    })
print(df.isna().sum())
print(df.dtypes)

print(f'variables categoricas: {df.select_dtypes("category").columns}')
print(f'variables numericas: {df.select_dtypes("number").columns}')
print(f'otras variables: {df.select_dtypes(exclude=["number", "category"]).columns}')

df = df.drop(columns=['id_ultimo_jefe', 'mes_incorporacion', 'tiempo_empresa', 'edad'])
df.to_parquet(f'../data/processed/{ds_type}_data.parquet', index=False,)

"""
OBS:
- Hay variables eticamente delicadas como el genero, estado civil
- El mes_incorporacion puede ser una categorica (como trimestre)
- Usando la variable id_ultimo_jefe podemos armar un grafo de jefes y subordinados
    - A partir del grafo podemos calcular la distancia entre jefes y subordinados
    - A partir del grafo podemos calcular la cantidad de subordinados de cada jefe
    - A partir del grafo podemos calcular la jerarquia de cada empleado
"""

df['mes_incorporacion'].plot(kind='hist', bins=4)

