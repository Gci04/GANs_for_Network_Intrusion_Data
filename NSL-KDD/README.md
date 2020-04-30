# NSL-KDD

implementation of an approach of generating artificial data to balance  **NLS-KDD** network Intrusion Benchmark dataset using Generative Adversarial Networks.

All the libraries can be pip installed

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. Install dependencies which are specified in requirements.txt. use `pip install -r requirements.txt` or `pip3 install -r requirements.txt`
1. Raw Data is being kept [here](Data) within this repo.
1. Nevigate to NSL-KDD directory : `cd NSL-KDD/`
1. Train the Generative model by running the command `python train.py` or `python3 train.py`
1. To test the performance run : `python test.py` of `python3 test.py` which will produce results and plots saved in : `imgs`, `nsl_pca_plots` & `results`

## Directory layout
    .
    ├── imgs                 # directory containing GAN training logs plots (accuracy, loss & KL-Divergence)
    ├── logs                 # deirectory containing raw GAN training logs
    ├── models               # Deep leaning and ML models implementations scripts
    ├── notebooks            # Jupyter notebooks related to EDA of NSL-KDD
    ├── nsl_pca_plots        # Visualizations of NSL-KDD data (PCA)
    ├── results              # ML models performances (accuracy, f1-score, precision & recall)
    ├── trained_generators   # Pretrained Generative models directory
    ├── utils                # helper methods for evaluations and data preprocessing
    ├── train.py             # Script to train ML models and Generative Adversarial Network
    ├── test.py              # Script to test performance of GAN Model approach
    └── README.md
## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

## Contact
If you would like to get in touch, please contact: [Gcinizwe Dlamini](google.com)

## References

1. M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

1. J. McHugh, “Testing intrusion detection systems: a critique of the 1998 and 1999 darpa intrusion detection system evaluations as performed by lincoln laboratory,” ACM Transactions on Information and System Security, vol. 3, no. 4, pp. 262–294, 2000.
