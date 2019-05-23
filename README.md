# Face Recognition

## About

### Assignment - Pattern Recognition CS 4044

Face Recognition system to distinguish features classes **0(Male)** 
and **1(Female)**.

Feature extraction approach used is PCA and LogisticRegression used for
classification algorithm to distinguishing the features.

### Students Details

- Name: Sangam Kumar
- Roll number: B150110CS

> Note: We credit [UNIVERSITY OF ESSEX](https://cswww.essex.ac.uk/mv/allfaces)
> for **Facial Images Dataset**.

### About Algorithms used

1. **Principal component analysis (PCA)**: Method used for feature extraction
is nothing but Linear dimensionality reduction using Singular Value
Decomposition of the data to project it to a lower dimensional space. It uses
the LAPACK implementation of the full SVD or a randomized truncated SVD by the
method of Halko et al. 2009, depending on the shape of the input data and the
number of components to extract. It can also use the scipy.sparse.linalg ARPACK
implementation of the truncated SVD. \
Implementation of the probabilistic PCA model from:
Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal component
analysis". Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 61(3), 611-622. via the score and score_samples methods. See
<http://www.miketipping.com/papers/met-mppca.pdf>.

2. **Logistic Regression classifier**: Logistic regression is a linear model
for classification rather than regression. Logistic regression is also known
in the literature as logit regression, maximum-entropy classification (MaxEnt)
or the log-linear classifier. In this model, the probabilities describing the
possible outcomes of a single trial are modeled using a logistic function. \
This implementation can fit binary, One-vs-Rest, or multinomial logistic
regression with optional *l*<sub>1</sub>, *l*<sub>2</sub> or Elastic-Net
regularization. Note that regularization is applied by default.

### Results

- > $ python train.py

  Results on test data:

  |               | precision     | recall        | f1-score      | support       |
  | :------------ | :-----------: | :-----------: | :-----------: | :-----------: |
  | 0             | 0.97          | 0.96          | 0.97          | 823           |
  | 1             | 0.81          | 0.84          | 0.82          | 157           |
  |               |               |               |               |               |
  | accuracy      |               |               | 0.94          | 980           |
  | macro avg     | 0.89          | 0.90          | 0.89          | 980           |
  | weighted avg  | 0.94          | 0.94          | 0.94          | 980           |

  The accuracy of classified data is 94%.

- > $ python test.py --img data/faces/men/male-700.jpg

  ```bash
  result: 0
  ```
  > $ python test.py --img data/faces/women/female-45.jpg

  ```bash
  result: 1
  ```
  > Note: 0 and 1 classes are for male and female respectively.

## Instructions

- [x] Install all the requirement using:
  > $ pip install -r requirement.txt
- [x] Train the data stored in `data/faces` using:
  > $ python train.py
- [x] Test any face image using:
  > $ python test.py --img image.jpg

### Advanced usage

`data/datas.json` contains mapping of face images stored in `data/faces`.

`config.py` contains the contant value of:
- **DATA_PATH**: Mapped `datas.json` file path
- **FEATURES_PATH**: Pickel file of features
- **REDUCTER_PATH**: Pickel file of reductores
- **MODEL_PATH**: Pickel file of model
- **TEST_PERCENTAGE**: Percentage of data to test on
- **N_DIMENSIONS**: Number of dimension to reduce to
- **IMG_SIZE**: Image dimension to generalise images
