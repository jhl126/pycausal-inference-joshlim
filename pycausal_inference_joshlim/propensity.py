# Solution obtained from Claude to solve for packaging error
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "packaging"], capture_output=True)

# Add imports
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from sklearn.linear_model import LinearRegression, LogisticRegression

"""Test 1: IPW with simple positive treatment effect"""
np.random.seed(42)
n = 1000        
# Generate data with known ATE = 2
x = np.random.normal(0, 1, n)
prob_t = 1 / (1 + np.exp(-(0.5 * x)))
t = np.random.binomial(1, prob_t, n)
y = 2 * t + x + np.random.normal(0, 0.5, n)       
df = pd.DataFrame({'x': x, 't': t, 'y': y})

"""Test 5: IPW with categorical covariate"""
np.random.seed(101)
n = 1000        
# Generate data with categorical confounder
group = np.random.choice(['A', 'B', 'C'], n)
group_effect = {'A': 0, 'B': 1, 'C': 2}
x_numeric = np.array([group_effect[g] for g in group])        
prob_t = 1 / (1 + np.exp(-(0.5 * x_numeric)))
t = np.random.binomial(1, prob_t, n)
y = 2.0 * t + x_numeric + np.random.normal(0, 0.5, n)        
df = pd.DataFrame({'group': group, 't': t, 'y': y})

def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    X = dmatrix(ps_formula, df)
    model = LogisticRegression(penalty = None, max_iter = 1000).fit(X, df[T])
    ps = model.predict_proba(X)[:,1]
    return np.mean((df[T] - ps) / (ps*(1-ps)) * df[Y])

def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    X = dmatrix(formula, df)
    model = LogisticRegression(penalty=None, max_iter=1000).fit(X,df[T])
    ps = model.predict_proba(X)[:,1]
    
    Y_mat, X_out = dmatrices(f"{Y} ~ {T} + {formula}", df)
    outcome_model = LinearRegression().fit(X_out, np.array(Y_mat).flatten())
    
    mu1_df = df.copy()
    mu0_df = df.copy()
    mu1_df[T] = 1
    mu0_df[T] = 0
    mu1x = dmatrix(f"{T} + {formula}", mu1_df)
    mu0x = dmatrix(f"{T} + {formula}", mu0_df)

    mu1 = outcome_model.predict(mu1x).flatten()
    mu0 = outcome_model.predict(mu0x).flatten()
    
    ate = np.mean(df[T] * (df[Y] - mu1) / ps + mu1) - np.mean((1-df[T]) * (df[Y] - mu0) / (1-ps) + mu0)
    return ate 