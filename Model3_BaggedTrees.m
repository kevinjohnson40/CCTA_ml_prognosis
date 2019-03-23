
function [Best_trainedClassifier,AUC_inner,hp_inner]=Model3_BaggedTrees(predictors,response,numfolds,numrepeats)

% Function for Bagged Trees model 
% CCTA data analysis
% Created 07-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised March 2019 to work inside B2_nested_CV.m 

% Grid search of hyperparameter combinations
% Returns the classifier with the best performance as judged by cross-validation
    
    
% Hyperparameters 
% % Nominal possible values:
%     [NumObservations,NumPredictors]=size(predictors);
%     NumLearningCycles={200 800 2000};
%     MinLeafSize={Integers logscaled range [1,floor(NumObservations/2)};
%     MaxNumSplits={integer logscaled range [1,NumObservations-1]};
%     NumVariablesToSample={Integers range [1,NumPredictors]};
% 
% % Actually searched: 
%     NumLearningCycles={200 1000};
%     MinLeafSize={1 10 100};
%     MaxNumSplits={1 10 100};
%     NumVariablesToSample={2 4 16};
    
% Final:
    NumLearningCycles={200};%200
    MinLeafSize={10};%10
    MaxNumSplits={100};%100
    NumVariablesToSample={1};%2
    
% Loops
    AUC=NaN(200,1);
    this_hpset=0;
    for hp1=1:length(MinLeafSize)
        minleaf=MinLeafSize{hp1};
    for hp2=1:length(MaxNumSplits)
        splits=MaxNumSplits{hp2};
    for hp3=1:length(NumVariablesToSample)
        nvs=NumVariablesToSample{hp3};
    for hp4=1:length(NumLearningCycles)
        cycles=NumLearningCycles{hp4};
        
        this_hpset=this_hpset+1;
    
        % Train a classifier
            template = templateTree('MaxNumSplits', splits,'NumVariablesToSample',nvs,'MinLeafSize',minleaf);
            classificationEnsemble = fitcensemble(...
                predictors, ...
                response, ...
                'Method', 'Bag', ...
                'NumLearningCycles', cycles, ...
                'Learners', template, ...
                'ClassNames', categorical({'0'; '1'}));

        % Create prediction function
            trainedClassifier.classificationEnsemble = classificationEnsemble;
            PredictFcn = @(x) predict(classificationEnsemble,x);
            trainedClassifier.predictFcn = @(x) PredictFcn(x);
            trainedClassifier.modelname='BaggedTrees';

        % Loop through repeated cross-validation
            AUC_repeat=nans(numrepeats,1);
            parfor repeat=1:numrepeats
                
                % Partition
                cv=cvpartition(response,'KFold',numfolds);
                partitionedModel = crossval(trainedClassifier.classificationEnsemble, 'CVpartition', cv);

                % Compute validation scores
                [~,validationscores] = kfoldPredict(partitionedModel);%first output is validation predictions
                scores=validationscores(:,2);

                % Find AUC as measure of performance
                [~,~,~,AUC_repeat(repeat)]=perfcurve(response,scores,1);

            end
            AUC(this_hpset)=nanmean(AUC_repeat);
            
        % Did this hyperparameter set give the best AUC? If so, update as current winner.
            maxsofar=nanmax(AUC);
            if maxsofar==AUC(this_hpset)
                Best_trainedClassifier=trainedClassifier;
                AUC_inner.AUC_mean=AUC(this_hpset);
                AUC_inner.AUC_distribution=AUC_repeat;
                hp_inner.NumLearningCycles=NumLearningCycles;
                hp_inner.MinLeafSize=minleaf;
                hp_inner.MaxNumSplits=splits;
                hp_inner.NumVariablesToSample=nvs;
                hp_inner.NumLearningCycles=cycles;
            end
    end
    end
    end
    end
    