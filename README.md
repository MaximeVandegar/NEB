![alt text](.imgs/header.jpg)


## Getting Started

```commandline
$ pip install -r requirements.txt
```

The implementation provides a simple interface that allows inference in a single line of code.
```python
from inference.neb import McBiasedEstimator, McUnbiasedEstimator, ElboEstimator, IwEstimator
# Define your data, likelihood function, source model, ...
estimator = McBiasedEstimator() 
estimator.infer(observations, source_model, optimizer, log_likelihood_fct)
```
The source code for the different estimators was written to be self-contained in a single file for a quick and easy understanding.

[Getting Started](https://github.com/MaximeVandegar/NEB/tree/main/examples)

## Cite

If you make use of this code in your work, please cite our paper:

```
@misc{vandegar2020neural,
      title={Neural Empirical Bayes: Source Distribution Estimation and its Applications to Simulation-Based Inference}, 
      author={Maxime Vandegar and Michael Kagan and Antoine Wehenkel and Gilles Louppe},
      year={2020},
}
```

