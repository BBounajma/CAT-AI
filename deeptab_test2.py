import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from deeptab.models import SAINTClassifier


model = joblib.load("saint_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")