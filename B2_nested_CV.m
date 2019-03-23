
function [final_classifier,ROC_outer]=B2_nested_CV(CCTAtable,modelname,redundancy,comment)

% CCTA data analysis
% Kevin M. Johnson, M.D. Yale University, March 2019

% This is an "outer" crossvalidation routine meant to
% measure the accuracy of the best classifiers.

% For each fold, it calls a "classifier creation algorithm (cca)" designated by "modelname"
% which finds the set of hyperparameters that best discriminates outcomes (responses)
% for that fold, as judged by a crossvalidation routine internal to the cca

% The cca returns a trained classifier that is then used to predict responses on
% the outer test fold.

% After all the folds have been run, overall accuracy is measured by ROC
% curve analysis, with area uder the curve (AUC) as the measure of
% accuracy (note: misclassifcation rate is not used).

% This procedure is repeated to give multiple estimates of AUC, from which a mean AUC and 
% its 95% confidence interval can be computed.

% The final trained classifier is made by repeating the training procedure
% within the cca using all of the data. This gives the hyperparameters
% described as "final" in the paper. 

% All predictors and responses
    predictors=table2array(CCTAtable(:,1:end-1));
    response=table2array(CCTAtable(:,end));
    
% Repeated outer CV
    KFolds = redundancy.numfolds_outer;
    AUC_fold=nans(KFolds,redundancy.numrepeats_outer);
    AUC_repeat=nans(redundancy.numrepeats_outer,1);
    for repeat=1:redundancy.numrepeats_outer
        
        % Stratified partition
        cvp = cvpartition(response, 'KFold', KFolds);

        % Perform "outer" cross-validation
        scores_folds=cell(KFolds,1);
        parfor fold = 1:KFolds
            disp([comment ' outer_repeat  ' num2str(repeat) '  Kfold ' num2str(fold)])
            
            % Training folds data
            x = predictors(cvp.training(fold),:);
            y = response(cvp.training(fold),:);

            % Call classifier creation algorithm for this fold
            if strcmp(modelname,'Model1_LogisticRegression')
                trainedClassifier=Model1_LogisticRegression(x,y,redundancy.numfolds_inner,redundancy.numrepeats_inner);
            elseif strcmp(modelname,'Model2_KNearestNeighbors')  
                trainedClassifier=Model2_KNearestNeighbors(x,y,redundancy.numfolds_inner,redundancy.numrepeats_inner);
            elseif strcmp(modelname,'Model3_BaggedTrees')  
                trainedClassifier=Model3_BaggedTrees(x,y,redundancy.numfolds_inner,redundancy.numrepeats_inner);
            elseif strcmp(modelname,'Model4_ClassificationNeuralNet')  
                trainedClassifier=Model4_ClassificationNeuralNet(x,y,redundancy.numfolds_inner,redundancy.numrepeats_inner); 
            end
            
            % Get predictive scores on test fold
            xtest = predictors(cvp.test(fold), :);
            [~,stest] = trainedClassifier.predictFcn(xtest);

            % Store scores in the original order
            scores_folds{fold} = stest(:,2);

            % Fold level: Area under ROC curve as measure of performance
            ytest = response(cvp.test(fold));
            [~,~,~,AUC_fold(fold,repeat)]=perfcurve(ytest,stest(:,2),1);

        end
        
        %kluge to get parfor loop to work in parallel above
        scores_all = NaN(size(response,1),redundancy.numrepeats_outer);
        for fold=1:KFolds 
            scores_all(cvp.test(fold),repeat)=scores_folds{fold};
        end
        
    end
    
% Compute and display AUC results
    AUC_fold_mean=nanmean(AUC_fold(:));
    AUC_fold_ci=bootci(100,@nanmean,AUC_fold(:));
    disp([comment, ' outer AUC with 95% confidence interval - repeat and fold'])
    disp(num2str([AUC_fold_mean;AUC_fold_ci])) 
    
% Store ROC information
    ROC_outer.AUC_fold_distribution=AUC_fold(:);
    ROC_outer.AUC_fold=[AUC_fold_mean;AUC_fold_ci];
    ROC_outer.response=response;
    scores=nanmean(scores_all,2);
    ROC_outer.scores=scores;
    [X,Y]=perfcurve(response,scores,1);
    ROC_outer.X=X;
    ROC_outer.Y=Y;
    ROC_outer.redundancy=redundancy;
    
% Make final classifier using all data - this AUC is overoptimistic
    xall = predictors;
    yall = response;
    [final_classifier,~,final_hp]=eval([modelname '(xall,yall,redundancy.numfolds_inner,redundancy.numrepeats_inner)']);
    final_classifier.final_hp=final_hp;
    final_classifier.modelname=modelname;
    
    