
%C_script_nn_classification.m

% CCTA data analysis
% Script for Neural Net Pattern Recognition 
% Created 12-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised 03-Jan-2019
%This implementation requires the MATLAB Parallel Computing Toolbox, but
%can be run without it by changing the "parfor" loops to "for" loops

%Three outcomes were defined: all deaths, coronary artery disease deaths,
%and CHD deaths plus myocardial infarctions. The variable numbering 1
%through 3 generallly refers to these three outcomes.

%Inputs: Features and outcomes are generated by the B_threshold_inputs.m function.
%Outputs: MLscores_NN1 are the machine learning scores
%         X_NN1 and Y_NN1 define the ROC curve for all deaths
%         AUC_NN1 is a 1x3 vector containing the AUCs, followed by the
%         pointwise bootstrap confidence interval upper and lower bounds

%Results vary somewhat from trial to trial because of randomization during the
%division of data within the nn algorithm into training, validation and test sets.

%This routine is time consuming. To run much faster, disable bootstrapping. 
%No confidence intervals will be returned in that case. This implementation 
%requires the MATLAB Parallel Computing Toolbox, but can be run without it by 
%changing "parfor" in the loop to "for".
    bootstrapping='on';
    
%load data
    load input_data/CCTAtable1.mat
    load input_data/CCTAtable2.mat
    load input_data/CCTAtable3.mat
    
    features1=table2array(CCTAtable1(:,1:end-1));
    features2=table2array(CCTAtable2(:,1:end-1));
    features3=table2array(CCTAtable3(:,1:end-1));
    outcome1=table2array(CCTAtable1(:,end));
    outcome2=table2array(CCTAtable2(:,end));
    outcome3=table2array(CCTAtable3(:,end));
    alldeaths=double(outcome1)-1;
    CHDdeaths=double(outcome2)-1;
    CHDMIdeaths=double(outcome3)-1;
    
% Run classification neural network for each outcome group
    [response_NN1,scores_NN1,X_NN1,Y_NN1,T1,AUC_NN1]=nn_classification(features1,alldeaths,bootstrapping); 
    [response_NN2,scores_NN2,X_NN2,Y_NN2,T2,AUC_NN2]=nn_classification(features2,CHDdeaths,bootstrapping); 
    [response_NN3,scores_NN3,X_NN3,Y_NN3,T3,AUC_NN3]=nn_classification(features3,CHDMIdeaths,bootstrapping); 
    
% Make results table
    NeuralNet_results_table=table(AUC_NN1',AUC_NN2',AUC_NN3');
    NeuralNet_results_table.Properties.VariableNames{'Var1'}='NeuralN_AUC_alldeaths';
    NeuralNet_results_table.Properties.VariableNames{'Var2'}='NeuralN_AUC_CHDdeaths';
    NeuralNet_results_table.Properties.VariableNames{'Var3'}='NeuralN_AUC_CHDdeathsplusMI';
    NeuralNet_results_table.Properties.VariableDescriptions={'NeuralN AUC followed by CI','AUC followed by CI','AUC followed by CI'};
    NeuralNet_results_table.Properties.UserData.Loops=100;
    NeuralNet_results_table.Properties.UserData.hiddenLayerSize=10;
    NeuralNet_results_table.Properties.UserData.trainFcn='trainscg';
    NeuralNet_results_table.Properties.UserData.performFcn='crossentropy';
    NeuralNet_results_table.Properties.UserData.percentile=parameter.percentile;
    NeuralNet_results_table.Properties.UserData
    disp(NeuralNet_results_table)
    NeuralNet_scores_and_outcomes=[scores_NN1 alldeaths scores_NN2 CHDdeaths scores_NN3 CHDMIdeaths];

% % Save
%     save('results/NeuralNet_results_table','NeuralNet_results_table')    
%     writetable(NeuralNet_results_table,'results/NeuralNet_results_table')
%     save('results/NeuralNet_scores_and_outcomes','NeuralNet_scores_and_outcomes')
%     csvwrite('results/NeuralNet_scores_and_outcomes.csv',NeuralNet_scores_and_outcomes,1)
    
    
function [response,MLscores,X_NN,Y_NN,T_NN,AUC_NN]=nn_classification(features,outcomes,bootstrapping)
    
%This code was derived from the "Neural Net Pattern Recogniton App (nprtool) in MATLAB

%feature normalization
    for cc=1:size(features,2)
        features(:,cc)=(features(:,cc)-nanmean(features(:,cc)))/nanstd(features(:,cc));
    end

% Loop
    o_all=nans(30,size(features,1));
    s_all=nans(30,size(features,1));
    parfor w=1:100

        % Inputs and outcomes
        x = features';
        o = outcomes';
        
        % Choose a training function
        % For a list of all training functions type "help nntrain"
        % 'trainlm' is usually fastest
        % 'trainbr' takes longer but may be better for challenging problems.
        % 'trainscg' uses less memory. Suitable in low memory situations.
        trainFcn = 'trainscg'; %trainscg

        % Create a pattern recognition network
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize, trainFcn);
        net.performFcn='crossentropy';

        % Set up division of data for training, validation, and testing
        [trainInd,valInd,testInd]=dividerand(length(o),0.40,0.10,0.50);
        net.divideFcn='divideind';
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = valInd;
        net.divideParam.testInd = testInd;

        % Train the network
        net = train(net,x,o);

        % Test the network
        x(:,trainInd)=NaN;
        x(:,valInd)=NaN;
        s = net(x);
        
        o_all(w,:)=o;
        s_all(w,:)=s;

    end
    
    %Find areas under ROC curves with pointwise confidence intervals
    outcomes=nanmean(o_all)';
    MLscores=nanmean(s_all)';
    if strcmp(bootstrapping,'on')
        [X_NN,Y_NN,T_NN,AUC_NN]=perfcurve(outcomes,MLscores,1,'nboot',1000);
    else
        [X_NN,Y_NN,T_NN,AUC_NN]=perfcurve(outcomes,MLscores,1);
    end
    response=outcomes;
end

