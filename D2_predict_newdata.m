
function [ROC_ind,response_predicted]=D2_predict_newdata(CCTAtable,trainedClassifier)

% CCTA data analysis
% Kevin M. Johnson, M.D. Yale University, March 2019

% Uses trained classifier to predict scores and responses on new data
% Since the actual responses are also known in a research setting, ROC 
% analysis can be done to find the accuracy of the classifier

% All predictors and responses
    predictors=table2array(CCTAtable(:,1:end-1));
    response_known=table2array(CCTAtable(:,end));

% Get predictions 
    [response_predicted,stest] = trainedClassifier.predictFcn(predictors);
    scores=stest(:,2);
    
% Area under ROC curve as measure of performance
    [X,Y,~,AUC]=perfcurve(response_known,scores,1);
        
% Store ROC information
    ROC_ind.AUC=AUC;
    ROC_ind.response=response_known;
    ROC_ind.scores=scores;
    ROC_ind.X=X;
    ROC_ind.Y=Y;
    