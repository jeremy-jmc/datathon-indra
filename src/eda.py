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


df_train = pd.read_parquet('../data/processed/train_data.parquet')
numerical_features = df_train.select_dtypes(include=[np.number]).columns
df_train

sns.pairplot(df_train[numerical_features], 
             hue='abandono_6meses', 
             diag_kind='kde'
             )



