from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Positive effect data
np.random.seed(42)
n = 1000
positive_effect_data = pd.DataFrame({
    'I': range(n),
    'T': np.random.binomial(1, 0.5, n),
    })
positive_effect_data['Y'] = np.where(
    positive_effect_data['T'] == 1, 
    np.random.normal(10, 2, n),
    np.random.normal(8, 2, n)
    )
        
# No effect data
np.random.seed(123)
n = 500
no_effect_data = pd.DataFrame({
    'I': range(n),
    'T': np.random.binomial(1, 0.5, n),
    })
no_effect_data['Y'] = np.random.normal(5, 3, n)

def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    avg_treatment = data.loc[data['T']==1]['Y'].mean()
    avg_control = data.loc[data['T']==0]['Y'].mean()
    ATE_estimate = avg_treatment - avg_control
    
    n_1 = len(data.loc[data['T']==1])
    n_0 = len(data.loc[data['T']==0])
    var_1 = data.loc[data['T']==1]['Y'].var()
    var_0 = data.loc[data['T']==0]['Y'].var()
    se_ate = np.sqrt(var_1/n_1 + var_0/n_0)
    
    z = norm.ppf(1-alpha/2)
    
    ci_lower = ATE_estimate - z*se_ate
    ci_upper = ATE_estimate + z*se_ate
    
    return(ATE_estimate, ci_lower, ci_upper)

def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    avg_treatment = data.loc[data['T']==1]['Y'].mean()
    avg_control = data.loc[data['T']==0]['Y'].mean()
    ATE_estimate = avg_treatment - avg_control
    
    n_1 = len(data.loc[data['T']==1])
    n_0 = len(data.loc[data['T']==0])
    var_1 = data.loc[data['T']==1]['Y'].var()
    var_0 = data.loc[data['T']==0]['Y'].var()
    se_ate = np.sqrt(var_1/n_1 + var_0/n_0)
    
    t_statistic = ATE_estimate / se_ate
    p_value = 2 * (1-norm.cdf(abs(t_statistic)))
    
    return(ATE_estimate, t_statistic, p_value) 