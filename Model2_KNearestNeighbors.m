
function [Best_trainedClassifier,AUC_inner,hp_inner]=Model2_KNearestNeighbors(predictors,response,numfolds,numrepeats)

% Function for K Nearest Neighbors model 
% CCTA data analysis
% Created 07-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised March 2019 to work inside B2_nested_CV.m 

% Grid search of hyperparameter combinations
% Returns the classifier with the best performance as judged by cross-validation
    
% Hyperparameters 
% % Nominal possible values:
%     Distance={'cosine' 'euclidean' 'cityblock' 'chebychev' 'minkowski'...
%         'correlation' 'spearman' 'hamming' 'jaccard' };
%     NumNeighbors={logscaled in the range [1, max(2,round(NumObservations/2))};
%     DistanceWeight={'equal' 'inverse' 'squaredinverse'};
% 
% % Actually searched: 
%     NumObservations=size(predictors,1);
%     Distance={'cosine' 'euclidean' 'cityblock'};
%     NumNeighbors={100 500 1000 2000};
%     DistanceWeight={'equal'};
    
% Final:
    Distance={'cosine'};
    NumNeighbors={1000};
    DistanceWeight={'equal'};
    
% Loops
    AUC=NaN(400,1);
    this_hpset=0;
    for hp1=1:length(Distance)
        dist=Distance{hp1};
    for hp2=1:length(NumNeighbors)
        numn=NumNeighbors{hp2};
    for hp3=1:length(DistanceWeight)
        wt=DistanceWeight{hp3};
 
        this_hpset=this_hpset+1;
        
        % Train a classifier
            classificationKNN = fitcknn(...
                predictors, ...
                response, ...
                'Distance', dist, ...
                'NumNeighbors', numn, ...
                'DistanceWeight', wt, ...
                'Standardize', 'on', ...
                'ClassNames', categorical({'0'; '1'}));
        
        % Create prediction function
            trainedClassifier.classificationKNN = classificationKNN;
            PredictFcn = @(x) predict(classificationKNN,x);
            trainedClassifier.predictFcn = @(x) PredictFcn(x);
            trainedClassifier.modelname='KNearestNeighbors';

        % Loop through repeated cross-validation
            AUC_repeat=nans(numrepeats,1);
            parfor repeat=1:numrepeats

                % Partition
                cv=cvpartition(response,'KFold',numfolds);
                partitionedModel = crossval(trainedClassifier.classificationKNN, 'CVpartition', cv);

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
                hp_inner.Distance=dist;
                hp_inner.NumNeighbors=numn;
                hp_inner.DistanceWeight=wt;
            end
    end
    end
    end
    
    