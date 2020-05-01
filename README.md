# DGM : A Data Generative Model to Improve Minority Classes Presence in Anomaly Detection Domain

This repository provides a Keras-Tensorflow implementation of an approach of generating artificial data to balance network Intrusion Benchmark datasets using Generative Adversarial Networks. Benchmarking datasets for Network Intrusion Detection : **NLS-KDD** and **UNSW-NB15**

## Prerequisites

* Keras >= 2.0.8
* TensorFlow >= 1.3.0
* Numpy >= 1.13.3
* Matplotlib >= 2.0.2
* Seaborn >= 0.7.1
* [tabulate 0.8.6](https://pypi.org/project/tabulate/)
* [imbalanced-learn](https://pypi.org/project/imbalanced-learn/)
* [Catboost](https://tech.yandex.com/catboost/)
* [category_encoders](http://contrib.scikit-learn.org/categorical-encoding/index.html)

All the libraries can be pip installed

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. Install dependencies which are specified in requirements.txt. use `pip install -r requirements.txt` or `pip3 install -r requirements.txt`
1. Raw Data is being kept [here](Data) within this repo.
1. Navigate to desired Data Generative Model `cd NSL-KDD` or `cd UNSW-NB15`, then train and test the model by running train and test scripts : `python train.py` first, then `python test.py`. 

## Repository directory layout

<!-- ### Repository directory layout -->

    .
    ├── Data                    # Benchmark datasets folder
    │   ├── NSL-KDD             # NLS-KDD Dataset folder
    │   ├── UNSW-NB15           # UNSW-NB15 Dataset folder
    │   └── README.md           # Dataset info
    ├── NSL-KDD                 # Implementation for NSL-KDD dataset
    │   ├── models              # Directory with implementation of the Generative Adversarial Networks and ML Classifiers
    │   ├── train.py            # Main file for testing model
    │   ├── test.py             # Main file for testing model
    │   ├── README.md           
    │   └── ...
    ├── UNSW-NB15               # Implementation for UNSW-NB15 dataset
    │   ├── models              # Directory with implementation of the Generative Adversarial Networks and ML Classifiers
    │   ├── train.py            # Main file for training and testing model
    │   ├── test.py             # Main file for testing model
    │   ├── README.md
    │   └── ...
    └── README.md

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

## Contact
If you would like to get in touch, please contact: <br/>
<!-- Gcinizwe Dlamini - g.dlamini@innopolis.university -->

## References
