import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression


def simple_data(self):
        """Generate simple data with known treatment effect"""
        np.random.seed(42)
        n = 1000
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Treatment assignment (confounded)
        prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        t = np.random.binomial(1, prob_t, n)
        
        # Outcome with constant treatment effect = 2.0
        y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
        
        # Split into train/test
        train = df.iloc[:800].copy()
        test = df.iloc[800:].copy()
        
        return train, test

def heterogeneous_data(self):
        """Generate data with heterogeneous treatment effect"""
        np.random.seed(123)
        n = 1500
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Treatment assignment
        prob_t = 1 / (1 + np.exp(-(0.4 * x1)))
        t = np.random.binomial(1, prob_t, n)
        
        # Outcome with heterogeneous effect: effect depends on x1
        # CATE(x1) = 1 + 0.5*x1
        te = 1.0 + 0.5 * x1
        y = te * t + x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
        
        train = df.iloc[:1200].copy()
        test = df.iloc[1200:].copy()
        
        return train, test

def continuous_treatment_data(self):
        """Generate data with continuous treatment"""
        np.random.seed(789)
        n = 1000
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Continuous treatment
        t = 10 + x1 + 2*x2 + np.random.normal(0, 1, n)
        
        # Outcome: linear effect of treatment
        y = t + x1 + 0.5*x2 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})
        
        train = df.iloc[:800].copy()
        test = df.iloc[800:].copy()
        
        return train, test

def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    model = LGBMRegressor()
    model.fit(train[X + [T]], train[y])
    
    t0 = test.copy()
    t1 = test.copy()
    
    t0[T] = 0
    t1[T] = 1
    
    cate = model.predict(t1[X + [T]]) - model.predict(t0[X + [T]])
    
    output = test.copy()
    output['cate'] = cate
    
    return output 

def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    t0 = train.loc[train[T] == 0]
    t1 = train.loc[train[T] == 1]
    
    model0 = LGBMRegressor()
    model1 = LGBMRegressor()
    
    model0.fit(t0[X], t0[y])
    model1.fit(t1[X], t1[y])
    
    cate = model1.predict(test[X]) - model0.predict(test[X])
    
    output = test.copy()
    output['cate'] = cate
    
    return output

def x_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    t0 = train.loc[train[T] == 0]
    t1 = train.loc[train[T] == 1]
    
    model0 = LGBMRegressor()
    model1 = LGBMRegressor()
    
    model0.fit(t0[X], t0[y])
    model1.fit(t1[X], t1[y])
    
    pseudo0 = model1.predict(t0[X]) - t0[y]
    pseudo1 = t1[y] - model0.predict(t1[X])
    
    tau_model0 = LGBMRegressor()
    tau_model1 = LGBMRegressor()
    
    tau_model0.fit(t0[X], pseudo0)
    tau_model1.fit(t1[X], pseudo1)
    
    lr = LogisticRegression(penalty=None)
    lr.fit(train[X], train[T])
    
    e = lr.predict_proba(test[X])[:, 1]
    
    cate = e * tau_model0.predict(test[X]) + (1 - e) * tau_model1.predict(test[X])
    
    output = test.copy()
    output['cate'] = cate
    
    return output

def double_ml_cate(train, test, X, T, y) -> pd.DataFrame:
    model_t = LGBMRegressor()
    model_t.fit(train[X], train[T])
    T_res = train[T] - model_t.predict(train[X])
    
    model_y = LGBMRegressor()
    model_y.fit(train[X], train[y])
    Y_res = train[y] - model_y.predict(train[X])
    
    Y_star = Y_res / T_res
    w = T_res ** 2
    
    model = LGBMRegressor()
    model.fit(train[X], Y_star, sample_weight = w)
    
    cate = model.predict(test[X])
    
    output = test.copy()
    output['cate'] = cate
    
    return output