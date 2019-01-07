
%Conventional_scores_regression.m

% CCTA data analysis
% Script to perform logistic regression on conventional vessel scores and outcomes
% Created 12-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised 03-Jan-2019

%Logistic regression is used to map conventional vessel scores to outcomes
%Inputs are prepared by B_feature_weight_thresholding.m
%Outputs are the area under the ROC curves (AUC) with point wise confidence intervals

%Three outcomes were defined: all deaths, coronary artery disease deaths,
%and CHD deaths plus myocardial infarctions. The variable numbering 1
%through 3 generallly refers to these three outcomes.

%This routine is time consuming. To run much faster, disable bootstrapping. 
%No confidence intervals will be returned in that case.
    bootstrapping='off';%'off' or 'on'

%load data
    load input_data/CCTAdataAugust2018.mat
    data_table=CCTAdataAugust2018;

%call routines to fit logistic regression
    outcomestring='inputTable.outcome1_death';
    [response_CADRADS1,scores_CADRADS1,X_CADRADS1,Y_CADRADS1,T_CADRADS1,AUC_CADRADS1] = trainClassifier(data_table,1,outcomestring,bootstrapping);
    [response_Leaman1,scores_Leaman1,X_Leaman1,Y_Leaman1,T_Leaman1,AUC_Leaman1] = trainClassifier(data_table,2,outcomestring,bootstrapping);
    [response_SPS1,scores_SPS1,X_SPS1,Y_SPS1,T_SPS1,AUC_SPS1] = trainClassifier(data_table,3,outcomestring,bootstrapping);
    [response_SSS1,scores_SSS1,X_SSS1,Y_SSS1,T_SSS1,AUC_SSS1] = trainClassifier(data_table,4,outcomestring,bootstrapping);
    [response_SIS1,scores_SIS1,X_SIS1,Y_SIS1,T_SIS1,AUC_SIS1] = trainClassifier(data_table,5,outcomestring,bootstrapping);

    outcomestring='inputTable.outcome2_CHDdeath';
    [response_CADRADS2,scores_CADRADS2,X_CADRADS2,Y_CADRADS2,T_CADRADS2,AUC_CADRADS2] = trainClassifier(data_table,1,outcomestring,bootstrapping);
    [response_Leaman2,scores_Leaman2,X_Leaman2,Y_Leaman2,T_Leaman2,AUC_Leaman2] = trainClassifier(data_table,2,outcomestring,bootstrapping);
    [response_SPS2,scores_SPS2,X_SPS2,Y_SPS2,T_SPS2,AUC_SPS2] = trainClassifier(data_table,3,outcomestring,bootstrapping);
    [response_SSS2,scores_SSS2,X_SSS2,Y_SSS2,T_SSS2,AUC_SSS2] = trainClassifier(data_table,4,outcomestring,bootstrapping);
    [response_SIS2,scores_SIS2,X_SIS2,Y_SIS2,T_SIS2,AUC_SIS2] = trainClassifier(data_table,5,outcomestring,bootstrapping);

    outcomestring='inputTable.outcome3_CHDdeath_or_MI';
    [response_CADRADS3,scores_CADRADS3,X_CADRADS3,Y_CADRADS3,T_CADRADS3,AUC_CADRADS3] = trainClassifier(data_table,1,outcomestring,bootstrapping);
    [response_Leaman3,scores_Leaman3,X_Leaman3,Y_Leaman3,T_Leaman3,AUC_Leaman3] = trainClassifier(data_table,2,outcomestring,bootstrapping);
    [response_SPS3,scores_SPS3,X_SPS3,Y_SPS3,T_SPS3,AUC_SPS3] = trainClassifier(data_table,3,outcomestring,bootstrapping);
    [response_SSS3,scores_SSS3,X_SSS3,Y_SSS3,T_SSS3,AUC_SSS3] = trainClassifier(data_table,4,outcomestring,bootstrapping);
    [response_SIS3,scores_SIS3,X_SIS3,Y_SIS3,T_SIS3,AUC_SIS3] = trainClassifier(data_table,5,outcomestring,bootstrapping);

% Make results table
    Conventional_results_table=table(AUC_CADRADS1',AUC_SSS1',AUC_SIS1',AUC_Leaman1',AUC_SPS1',...
        AUC_CADRADS2',AUC_SSS2',AUC_SIS2',AUC_Leaman2',AUC_SPS2',...
        AUC_CADRADS3',AUC_SSS3',AUC_SIS3',AUC_Leaman3',AUC_SPS3');
    Conventional_results_table.Properties.VariableNames{'Var1'}='AUC_CADRADS1';
    Conventional_results_table.Properties.VariableNames{'Var2'}='AUC_SSS1';
    Conventional_results_table.Properties.VariableNames{'Var3'}='AUC_SIS1';
    Conventional_results_table.Properties.VariableNames{'Var4'}='AUC_Leaman1';
    Conventional_results_table.Properties.VariableNames{'Var5'}='AUC_SPS1';
    Conventional_results_table.Properties.VariableNames{'Var6'}='AUC_CADRADS2';
    Conventional_results_table.Properties.VariableNames{'Var7'}='AUC_SSS2';
    Conventional_results_table.Properties.VariableNames{'Var8'}='AUC_SIS2';
    Conventional_results_table.Properties.VariableNames{'Var9'}='AUC_Leaman2';
    Conventional_results_table.Properties.VariableNames{'Var10'}='AUC_SPS2';
    Conventional_results_table.Properties.VariableNames{'Var11'}='AUC_CADRADS3';
    Conventional_results_table.Properties.VariableNames{'Var12'}='AUC_SSS3';
    Conventional_results_table.Properties.VariableNames{'Var13'}='AUC_SIS3';
    Conventional_results_table.Properties.VariableNames{'Var14'}='AUC_Leaman3';
    Conventional_results_table.Properties.VariableNames{'Var15'}='AUC_SPS3';
    disp(Conventional_results_table)
    
% Save
    save('results/Conventional_results_table','Conventional_results_table')
    writetable(Conventional_results_table,'results/Conventional_results_table')
    
function [response,scores,X_LR,Y_LR,T_LR,AUC_LR] = trainClassifier(trainingData,ss,outcomestring,bootstrapping)

%This code was adapted from the MATLAB Classification Learner App code generator

% Extract predictors and response
    inputTable = trainingData;
    predictorNames = {'CADRADS', 'LeS', 'SPS03', 'SSS', 'SIS'};
    predictors = inputTable(:, predictorNames);
    response = categorical(eval(outcomestring));

% Data transformation: Select subset of the features
    vec=[false false false false false];
    vec(ss)=true;
    includedPredictorNames = predictors.Properties.VariableNames(vec);
    predictors = predictors(:,includedPredictorNames);

% Train a classifier
    successClass = '1';
    failureClass = '0';

% Compute the majority response class
    numSuccess = sum(response == successClass);
    numFailure = sum(response == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    responseCategories = {successClass, failureClass};
    successFailureAndMissingClasses = categorical({successClass; failureClass; missingClass}, responseCategories);
    isMissing = isundefined(response);
    zeroOneResponse = double(ismember(response, successClass));
    zeroOneResponse(isMissing) = NaN;
    
% Prepare input arguments to fitglm.
    concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];
    
% Train using fitglm.
    GeneralizedLinearModel = fitglm(concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial','link', 'logit');

% Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [p, 1-p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    featureSelectionFcn = @(x) x(:,includedPredictorNames);
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(featureSelectionFcn(predictorExtractionFcn(x)));

% Extract predictors and response
    inputTable = trainingData;
    predictorNames = {'CADRADS', 'LeS', 'SPS03', 'SSS', 'SIS'};
    predictors = inputTable(:, predictorNames);
    response = categorical(eval(outcomestring));

% Perform cross-validation
    KFolds = 10;
    cvp = cvpartition(response, 'KFold', KFolds);
    
% Initialize the predictions to the proper sizes
    validationPredictions = response;
    numObservations = size(predictors, 1);
    numClasses = 2;
    validationScores = NaN(numObservations, numClasses);
    for fold = 1:KFolds
        trainingPredictors = predictors(cvp.training(fold), :);
        trainingResponse = categorical(response(cvp.training(fold), :));

        % Data transformation: Select subset of the features
        % This code selects the same subset of features as were used in the app.
            vec=[false false false false false];
            vec(ss)=true;
            includedPredictorNames = trainingPredictors.Properties.VariableNames(vec);
            trainingPredictors = trainingPredictors(:,includedPredictorNames);

        % Train a classifier
            successClass = '1';
            failureClass = '0';
            
        % Compute the majority response class. If there is a NaN-prediction from
        % fitglm, convert NaN to this majority class label.
            numSuccess = sum(trainingResponse == successClass);
            numFailure = sum(trainingResponse == failureClass);
            if numSuccess > numFailure
                missingClass = successClass;
            else
                missingClass = failureClass;
            end
            responseCategories = {successClass, failureClass};
            successFailureAndMissingClasses = categorical({successClass; failureClass; missingClass}, responseCategories);
            isMissing = isundefined(trainingResponse);
            zeroOneResponse = double(ismember(trainingResponse, successClass));
            zeroOneResponse(isMissing) = NaN;
            
        % Prepare input arguments to fitglm.
            concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
            
        % Train using fitglm.
            GeneralizedLinearModel = fitglm(...
                concatenatedPredictorsAndResponse, ...
                'Distribution', 'binomial', ...
                'link', 'logit');

        % Convert predicted probabilities to predicted class labels and scores.
            convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
            returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
            scoresFcn = @(p) [p, 1-p];
            predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

        % Create the result struct with predict function
            featureSelectionFcn = @(x) x(:,includedPredictorNames);
            logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x));
            validationPredictFcn = @(x) logisticRegressionPredictFcn(featureSelectionFcn(x));

        % Compute validation predictions
            validationPredictors = predictors(cvp.test(fold),:);
            [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);

        % Store predictions in the original order
            validationPredictions(cvp.test(fold), :) = foldPredictions;
            validationScores(cvp.test(fold), :) = foldScores;           
    end
    scores=validationScores(:,1);
        
    % ROC curves with bootstrap confidence intervals
    if strcmp(bootstrapping,'on')
        [X_LR,Y_LR,T_LR,AUC_LR]=perfcurve(response,scores,1,'nboot',1000);
    else
        [X_LR,Y_LR,T_LR,AUC_LR]=perfcurve(response,scores,1);
    end
    
end
    
