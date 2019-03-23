
function [trainedClassifier,AUC_inner,hp_inner] = Model1_LogisticRegression(predictors,response,numfolds,numrepeats)

% Function for Logistic Regression with cross-validation
% CCTA data analysis
% Created 13-Mar-2019 Kevin M. Johnson, M.D. Yale University
% Adapted from output of MATLAB Classification Learner App

% Extract predictors and response
    predictors = array2table(predictors);

% Prepare response
    successClass = double(1);
    failureClass = double(0);
    numSuccess = sum(response == successClass);
    numFailure = sum(response == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    successFailureAndMissingClasses = [successClass; failureClass; missingClass];
    isMissing = isnan(response);
    zeroOneResponse = double(ismember(response, successClass));
    zeroOneResponse(isMissing) = NaN;

% Prepare input arguments to fitglm.
    concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];

% Train using fitglm.
    GeneralizedLinearModel = fitglm(concatenatedPredictorsAndResponse,...
        'Distribution', 'binomial','link', 'logit');

% Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [1-p, p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(x);

% Add additional fields to the result struct
    trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
    trainedClassifier.SuccessClass = successClass;
    trainedClassifier.FailureClass = failureClass;
    trainedClassifier.MissingClass = missingClass;
    trainedClassifier.ClassNames = {successClass; failureClass};
    trainedClassifier.modelname='LogisticRegression';

% Loop through repeated cross-validation
    AUC_repeat=nans(numrepeats,1);
    for repeat=1:numrepeats
        KFolds = numfolds;
        cvp = cvpartition(response, 'KFold', KFolds);

        % Cross-validation
        validationPredictions = response;
        numObservations = size(predictors, 1);
        numClasses = 2;
        validationScores = NaN(numObservations, numClasses);
        for fold = 1:KFolds
            
            % Predictors and response
            trainingPredictors = predictors(cvp.training(fold), :);
            trainingResponse = response(cvp.training(fold), :);

            % Train a classifier
            successClass = double(1);
            failureClass = double(0);

            % Compute the majority response class. 
            numSuccess = sum(trainingResponse == successClass);
            numFailure = sum(trainingResponse == failureClass);
            if numSuccess > numFailure
                missingClass = successClass;
            else
                missingClass = failureClass;
            end
            successFailureAndMissingClasses = [successClass; failureClass; missingClass];
            isMissing = isnan(trainingResponse);
            zeroOneResponse = double(ismember(trainingResponse, successClass));
            zeroOneResponse(isMissing) = NaN;

            % Prepare input arguments to fitglm.
            concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];

            % Train using fitglm.
            GeneralizedLinearModel = fitglm(concatenatedPredictorsAndResponse, ...
                'Distribution', 'binomial','link', 'logit');

            % Convert predicted probabilities to predicted class labels and scores.
            convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
            returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
            scoresFcn = @(p) [1-p, p];
            predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

            % Create the result struct with predict function
            logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
            validationPredictFcn = @(x) logisticRegressionPredictFcn(x);

            % Compute validation predictions and scores
            validationPredictors = predictors(cvp.test(fold), :);
            [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);

            % Store predictions and scores in the original order
            validationPredictions(cvp.test(fold), :) = foldPredictions;
            validationScores(cvp.test(fold), :) = foldScores;
        end
        scores=validationScores(:,2);
        
        % Find AUC as measure of performance 
        [~,~,~,AUC_repeat(repeat)]=perfcurve(response,scores,1);
        
    end
    
% Output
    AUC_inner.AUC_mean=nanmean(AUC_repeat);
    AUC_inner.AUC_distribution=AUC_repeat;
    hp_inner=[];
    
