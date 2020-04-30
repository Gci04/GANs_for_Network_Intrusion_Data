# Network Intrusion Benchmark Datasets
Two datasets are contained in this folder : <br>
1. **[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)** : data set suggested to solve some of the inherent problems of the KDD'99 data set. Although, this new version of the KDD data set still suffers from some of the problems discussed by McHugh and may not be a perfect representative of existing real networks.   

1. **[UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)** : Network Intrusion data created by the Cyber Range Lab of the Australian Centre for Cyber Security. They captured the data by using the tool tcpdump.
It contains real network traffic of normal patterns and
synthetic anomalies of network traffic.


## Directory layout
    .
    ├─ NSL-KDD                            # NLS-KDD Dataset folder
    |     ├─ KDDTest.csv                  # Test dataset
    |     ├─ KDDTrain.csv                 # Train dataset
    |     └─ nslkdd_profile.html          # Train dataset summary
    |
    ├── UNSW-NB15                         # UNSW-NB15 Dataset folder
    |     ├─ UNSW_NB15_testing-set.csv    # Test dataset
    |     ├─ UNSW_NB15_training-set.csv   # Train dataset
    |     └─ unsw_profile.html          # Train dataset summary
    |
    └── README.md    


## References

1. Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.

1. M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

1. J. McHugh, “Testing intrusion detection systems: a critique of the 1998 and 1999 darpa intrusion detection system evaluations as performed by lincoln laboratory,” ACM Transactions on Information and System Security, vol. 3, no. 4, pp. 262–294, 2000.
