[![Tests](https://github.com/jhl126/pycausal-inference-joshlim/workflows/Tests/badge.svg)](https://github.com/jhl126/pycausal-inference-joshlim/actions)

[![PyPI version](https://badge.fury.io/py/pycausal-inference-joshlim.svg)](https://pypi.org/project/pycausal-inference-joshlim/)

# Causal Inference Python Package - Josh Lim

This package provides key causal inference methods. These methods include ATE estimation from randomized experiments, propensity score methods, and meta-learners.

## Installation

```bash
# Clone the repository
git clone https://github.com/jhl126/pycausal-inference-joshlim.git
cd pycausal-inference-joshlim

# Install in editable mode
uv pip install -e .
```

## Usage

Import functions with the following code:

```python
from pycausal_inference_joshlim import calculate_ate_ci, calculate_ate_pvalue
from pycausal_inference_joshlim import ipw, doubly_robust
from pycausal_inference_joshlim import s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate
```

## API Documentation

### RCT Module
- `calculate_ate_ci(data)` - Calculates the average treatment effect (ATE) and confidence interval from randomized experiment data
- `calculate_ate_pvalue(data)` - Calculates the p-value for the ATE estimate

### Propensity Score Module
- `ipw(data)` - Estimates the ATE using inverse probability weighting
- `doubly_robust(data)` - Estimates the ATE using the doubly robust estimator

### Meta-Learners Module
- `s_learner_discrete(data)` - Estimates heterogeneous treatment effects using the S-Learner approach
- `t_learner_discrete(data)` - Estimates heterogeneous treatment effects using the T-Learner approach
- `x_learner_discrete(data)` - Estimates heterogeneous treatment effects using the X-Learner approach
- `double_ml_cate(data)` - Estimates heterogeneous treatment effects using Double ML