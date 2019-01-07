# CCTA_ml_prognosis
Machine Learning and Coronary CT Angiograms

Date: January 7, 2019
From: Kevin M. Johnson, M.D.
Re: Machine Learning Analysis in MATLAB

This README describes practical aspects of data analysis for the original research paper
“Using Machine Learning to Find  Optimal Prognostic Plaque Characteristics on Coronary CT Angiograms” by Johnson KM et al, under review by the journal Radiology, manuscript RAD-18-2061. It is essentially identical to the Addendum submitted with that paper.

The topics are as follows:

1. Purpose of the research
2. Feature selection using the ReliefF algorithm 
3. MATLAB Classification Learner App overview
4. Links to documentation for models
5. How cross validation was handled
6. Link to Github repository
7. Description of the input data file
8. Description of the MATLAB scripts and functions

The information in quotes is abstracted from MATLAB documentation. Some references also appear in the main manuscript but are relisted here for convenience.


Purpose of the Research

Coronary artery disease severity is depicted on CT angiograms; certain vessel features such as stenoses, atherosclerotic plaque amount, etc. can be described by the observer. The question of interest is how to combine these features together to best predict the risk for a future event such as death or myocardial infarction. The dataset contains the vessel imaging features and outcomes for 6892 patients followed for a median of 9 years. Multiple machine learning models were tested by the authors for their discriminatory value. The data and MATLAB scripts and functions have been uploaded to Github were they can be downloaded and explored by any user (MATLAB, Statistics and Machine Learning Toolbox, and the Neural NetworkToolbox are required. Parallel Computing Toolbox recommended).


Feature selection using the ReliefF algorithm
As a preliminary step, an algorithm is applied to determine which input variables are most likely to be of predictive use [1,2]. ReliefF “finds the weights of predictors in the case where [the outcome] is a multiclass categorical variable. The algorithm penalizes the predictors that give different values to neighbors of the same class, and rewards predictors that give different values to neighbors of different classes. ReliefF first sets all predictor weights Wj to 0. Then, the algorithm iteratively selects a random observation xr, finds the k-nearest observations to xr for each class, and updates, for each nearest neighbor xq, all the weights for the predictors…” [3]
1. Kira K, Rendell LA. The feature selection problem: traditional methods and a new algorithm. American Association Artificial Intelligence. AAAI-92 Proceedings, Tenth National Conference on Artificial Intelligence 1992;  pp. 129-134.

2. Robnik-Sikonja M, Kononenko I. Theoretical and empirical analysis of ReliefF and RReliefF. Machine Learning 2003; 53: 23-69. 

3.  https://www.mathworks.com/help/stats/relieff.html


MATLAB Classification Learner App overview

The MATLAB Classification Learner App is a feature of the Statistics and Machine Learning Toolbox. A graphical user interface provides the means to analyze data using many different model types, and to set parameters as desired for each model. For the present paper, many permutations were explored, only some of which were ultimately included in the results table. Many more permutations are possible. The App can generate code for adaptation by the user to perform even more complex analyses. 

https://www.mathworks.com/products/statistics/classification-learner.html

In MATLAB, the classification neural network calculations are implemented via the Neural Network toolbox rather than wthin the Classification Learner App.
 

Link to documentation for models

The model types employed included decision trees, discriminant analysis, logistic regression, support vector machines, nearest neighbor classifiers, boosted trees, bagged trees, and classification neural networks. Supplemental Table 6 provides detail on parameters used, and the results. The following provides further links to detailed documentation:

https://www.mathworks.com/help/stats/classification.html


How cross validation was handled

Neural networks for classification averaged results over 30 iterations using different random divisions of data into training, validation and testing sets each time. The other models used tenfold cross validation: 

 https://www.mathworks.com/discovery/cross-validation.html


Link to Github repository

The data, scripts and functions can be found at:

https://github.com/kevinjohnson40/CCTA_ml_prognosis/



Description of the input data file

The coronary arterial tree is considered to be comprised of 16 segments; for example, the proximal third of the right coronary is one segment, the distal third of the left anterior descending coronary is another, etc. For each segment, 4 features are defined: degree of stenosis, amount of plaque, amount of calcification, and presence of remodeling (focal external diameter dilatation). Thus there are 64 imaging features per patient. Each feature is assigned a value depending on the degree of abnormality. The overall score for a given patient is the sum of these values.

CCTAdataAugust2018.mat is a 6892 x 72 table; the first 64 columns are the coronary artery segment features, columns 65 through 69 are conventional scores, and columns 70 through 72 are outcomes. 

Conventional scores are the reference standard against which the machine learning results are to be compared. There are 5 such scores: CAD-RADS, LeS, SPS, SSS, and SIS. See the Methods section of the paper for details.

Outcomes are the events that occur in the years of follow-up after the CT scan has been performed. Three events have been defined: all deaths, coronary artery deaths, and the sum of coronary artery deaths and myocardial infarctions. Median follow-up was 9.0 years. In general, variable and data numbering in the MATLAB code refer to these as 1 through 3 respectively.



Description of the MATLAB scripts and functions

A_feature_selection.m - This is the first routine to run. It loads the data, then performs a univariate analysis to determine which vessel features contribute appreciably to discrimination of events (outcomes). The output is a table with the features sorted by weight from highest to lowest for each class of outcome. 

B_feature_weight_thresholding.m - This is the second routine to run. It applies a user-selected percentile threshold to the feature weights in order to eliminate from further consideration those features that have little influence on the outcomes. The output is three tables, one for each outcome class, suitable for input into the MATLAB Classification Learner and neural network algorithms. For each table the first columns are the (retained) vessel features, and the last column is the outcome.

Conventional_scores_regression.m - This runs logistic regression on each of the conventional vessel scores versus each outcome. The output is a table of areas under the receiver operating characteristic curve (AUC).  These are a measure of the ability of the scores to discriminate those patient who subsequently had events from those who did not. These AUC values are the reference against which the machine-learning derived AUC values will be compared.

The machine learning models are in the following scripts:

Model1_DecisionTrees.m
Model2_LinearDiscriminant.m
Model3_LogisticRegression.m
Model4_SupportVectors.m
Model5_NearestNeighbors.m
Model6_BoostedTrees.m
Model7_BaggedTrees.m
Model8_ClassificationNeuralNet.m

The output of each routine is a table of AUCs for each of the three outcome classes, with pointwise confidence intervals. Modeling parameters can be selected for many of these; see Link to documentation for models above.  The paper lists the parameters used by the authors in Supplemental Table 6. It should be kept in mind that the output varies slightly with each run of these models because of the cross-validation procedure, and in some cases because of other uses of random sampling within the routines.

