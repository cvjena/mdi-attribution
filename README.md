# Anomaly Attribution of Multivariate Time-Series using Counterfactual Reasoning

This repository contains the source code for the method described in the following publication:

> [**Anomaly Attribution of Multivariate Time Series using Counterfactual Reasoning.**][1]  
> Violeta Teodora Trifunov and Maha Shadaydeh and Björn Barz and Joachim Denzler.  
> *20th IEEE International Conference on Machine Learning and Applications (ICMLA)*, 2021.

An example of how the counterfactual replacements can be used for anomaly attribution is provided in the Jupyter notebook [`MDI Attribution.ipynb`](MDI%20Attribution.ipynb).


## Requirements

The replacement algorithm itself requires `numpy` and `scipy`.

For the anomaly detection using the MDI algorithm and re-scoring of replaced intervals for attribution, the [`libmaxdiv`][2] library needs to be installed.
Please refer to the README in that repository for instructions on how to compile `libmaxdiv`.


[1]: https://arxiv.org/pdf/2109.06562
[2]: https://cvjena.github.io/libmaxdiv/