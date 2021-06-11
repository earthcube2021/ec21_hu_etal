# Machine Learning Guided Weather Analogs

_EarthCube 2021 Annual Meeting Presentation_

## Abstract

The Analog Ensemble (AnEn) is a computationally efficient technique for generating ensembles without requiring multiple simulation runs. The weather analogs are sought for a fine geographic scale and within a short time window, and have been shown to be calibrated and accurate.

The core of the AnEn is a metric that quantifies the similarity between current and historical forecasts, and it has as yet been a normalized temporal Euclidean distance based on multivariate forecasts variables. However, forecast variables are usually highly correlated and therefore it is paramount to weight the importance of each variable.  Brute force algorithms are usually employed, which are computationally inefficient and might fail to capture non-linear relationships among variables.

This notebook demonstrates Deep Analogs (DA), an implementation of AnEn which employs a Machine Learning (ML) based similarity metric. DA applies Machine Learning methods to improve AnEn and to cope with a much larger search space. First, an embedding network is trained to transform weather variables, and then a similarity is computed in latent space. DA is shown to outperform the Euclidian based AnEn in terms of prediction error and also in its ability to cope with large forecast errors. This notebook aims to showcase the latest research progress in DA and to contribute to bridging weather forecasting with Machine Learning research.

## About This Repository

This repository is created for the presentation at the [2021 EarthCube Annual Meeting](https://web.cvent.com/event/6589b2a2-9fd5-4e0b-a214-e0ba1c6348fe/summary) during June 15 - 17, 2021.

The presenter is [Weiming Hu](https://weiming-hu.github.io/).

Please find more information below:

1. On Analog Ensemble:
    1. [Parallel Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/)
    1. [RAnEn](https://weiming-hu.github.io/AnalogsEnsemble/R/) and [RAnEnExtra](https://weiming-hu.github.io/RAnEnExtra/)
    1. [PyAnEn](https://github.com/Weiming-Hu/PyAnEn)
1. On [Deep Analog](https://github.com/Weiming-Hu/DeepAnalogs)

## How to Run

You can directly view the content of the notebook by clicking [WH_01_Spatio-Temporal_Deep_Analogs_2021](WH_01_Spatio-Temporal_Deep_Analogs_2021.ipynb).

Or you can start an **interactive** Jupyter notebook session by clicking the below icon. You will be redirected to Binder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Weiming-Hu/EarthCube2021/HEAD?filepath=WH_01_Spatio-Temporal_Deep_Analogs_2021.ipynb)

## Questions

If you have any questions, please feel free to open an ticket or find Weiming [here](https://weiming-hu.github.io/).
