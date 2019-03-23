
function [Best_trainedClassifier,AUC_inner,hp_inner]=Model4_ClassificationNeuralNet(predictors,response,numfolds,numrepeats)

% Function for Classification Neural Network 

% CCTA data analysis
% Created 07-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised March 2019 to work inside B2_nested_CV.m 

% Grid search of hyperparameter combinations
% Returns the classifier with the best performance as judged by cross-validation

% Hyperparameters 
% % Nominal possible values:
%     trainFcn = {'trainscg' 'trainlm' 'trainbr' (and others)};
%     hiddenLayerSize = {integer typically >=5};
%     performFcn={'crossentropy' 'mse' 'msereg'};

% % Actually searched: 
%     trainFcn = {'trainscg' 'trainlm'};
%     hiddenLayerSize = {5 10 15 20};
%     performFcn={'crossentropy' 'mse' 'msereg'};

% Final:
    trainFcn = {'trainscg'}; 
    hiddenLayerSize = {15};
    performFcn={'crossentropy'};
    
% Loops
    AUC=NaN(400,1);
    this_hpset=0;
    for hp1=1:length(trainFcn)
        trainF=trainFcn{hp1};
    for hp2=1:length(hiddenLayerSize)
        hiddenL=hiddenLayerSize{hp2};
    for hp3=1:length(performFcn)
        performF=performFcn{hp3};
        if strcmp(trainF,'trainlm')&&strcmp(performF,'crossentropy')
            performF='msereg';
        end
 
        this_hpset=this_hpset+1;
        
        % Train on all data
        
            % All data
            x = predictors';
            y = response';

            % Create a pattern recognition network
            net = patternnet(hiddenL, trainF);
            net.performFcn=performF;
            net.trainParam.showWindow=0;

            % Set up division of data for training, validation, and testing
            net.divideFcn = 'dividerand';  % Divide data randomly
            net.divideParam.trainRatio = 85/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 0/100;

            % Train the network
            net_trained = train(net,x,y);
            
            % Create prediction function
            trainedClassifier.predictFcn = @(x) netplus(net_trained,x);
            trainedClassifier.modelname='ClassificationNeuralNet';

        % Do repeated KFold cross-validation
            scores_repeated = NaN(size(response,1),numrepeats);
            AUC_repeat=NaN(numrepeats,1);
            for repeat=1:numrepeats
            
                % Stratified partition
                cvp = cvpartition(response, 'KFold', numfolds);

                % Cross-validation
                scores_folds=cell(numfolds,1);
                parfor fold = 1:numfolds

                    % Train on numfolds-1 folds
                        x = predictors(cvp.training(fold), :)';
                        y = response(cvp.training(fold), :)';

                    % Create a pattern recognition network
                        net = patternnet(hiddenL, trainF);
                        net.performFcn=performF;
                        net.trainParam.showWindow=0;

                    % Set up division of data for training, validation, and testing
                        net.divideFcn = 'dividerand';  % Divide data randomly
                        net.divideParam.trainRatio = 85/100;
                        net.divideParam.valRatio = 15/100;
                        net.divideParam.testRatio = 0/100;

                    % Train the network
                        net_trained = train(net,x,y);

                    % Get scores on test fold
                        xval = predictors(cvp.test(fold), :)';
                        sval = net_trained(xval); 

                    % Store scores in the original order
                    scores_folds{fold} = sval;
                end
                for fold=1:numfolds %required kluge to get parfor to work above
                    scores_repeated(cvp.test(fold),repeat)=scores_folds{fold};
                end

                % Find AUC as measure of performance 
                [~,~,~,AUC_repeat(repeat)]=perfcurve(response,scores_repeated(:,repeat),1);
                
            end
            AUC(this_hpset)=nanmean(AUC_repeat);
        
        % Did this hyperparameter set give the best AUC? If so, update as current winner.
            maxsofar=nanmax(AUC);
            if maxsofar==AUC(this_hpset)
                Best_trainedClassifier=trainedClassifier;
                AUC_inner.AUC_mean=AUC(this_hpset);
                AUC_inner.AUC_distribution=AUC_repeat;
                hp_inner.Properties.VariableNames={'trainF' 'hiddenL' 'performF'};
                hp_inner.trainFcn=trainF;
                hp_inner.hiddenLayerSize=hiddenL;
                hp_inner.performFcn=performF;
            end
            
    end
    end
    end

function [response_predicted,scores]=netplus(net,x)

% Ensures compatability of predictFcn output with those of other model algorithms
% The predicted responses are not coded here - in practice, these depend on
% the score cutoff selected. We use area under the ROC as the measure instead
% of misclassification rate.

scores=NaN(size(x,1),2);
temp=net(x');
scores(:,2)=temp';
scores(:,1)=1-scores(:,2);
response_predicted=[]; 
