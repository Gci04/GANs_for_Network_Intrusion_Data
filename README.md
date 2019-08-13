# GANs for Improving Network Intrusion Benchmark Datasets

This repository provides a Keras-Tensorflow implementation of an approach of generating articial data to balance network Intrusion Benchmark datasets using Generative Adversarial Networks. Benchmarking datasets for Network Intrusion Detection : **KDD-99**, **NLS-KDD** and **UNSW-NB15**

## Dependencies and necessary libraries

## Repository directory layout

### Repository directory layout

    .
    ├── Data                 # Benchmark datasets folder
    │   ├── NSL-KDD          # NLS-KDD Dataset folder
    │   ├── UNSW-NB15        # UNSW-NB15 Dataset folder
    │   └── KDD-99           # NLS-KDD Dataset folder
    ├── NSL-KDD                 # Implementation for NSL-KDD dataset
    │   ├── preprocessing.py    # Data reprocessing file
    │   ├── main.py             # Main file for training and testing model
    │   ├── classifiers.py      # file containing the implementation the ML classification models
    │   ├── Utils.py            # File with implementation of the Generative Adversarial Networks
    │   └── ...
    ├── UNSW-NB15               # Implementation for UNSW-NB15 dataset
    │   ├── preprocessing.py    # Data reprocessing file
    │   ├── main.py             # Main file for training and testing model
    │   ├── classifiers.py      # file containing the implementation the ML classification models
    │   ├── Utils.py            # File with implementation of the Generative Adversarial Networks
    │   └── ...
    └── README.md

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

## Contact
If you would like to get in touch, please contact: <br/>
<!-- Gcinizwe Dlamini - g.dlamini@innopolis.university -->
