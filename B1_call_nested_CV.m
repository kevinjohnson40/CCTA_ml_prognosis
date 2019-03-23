
% Script - B1_call_nested_CV

% CCTA data analysis
% Kevin M. Johnson, M.D. Yale University, March 2019

% Prepares data and calls nested crossvalidation routine
% Multiple types of model algorithms can be called, see list below

% Input is patient data loaded and prepared by A_load_data.m.
% Initial columns are vessel features as predictors, and the last three columns are outcomes 

% Three outcomes are: all deaths, coronary artery disease deaths,
% and CHD deaths plus myocardial infarctions. 

% Load data
    load input_data/data_table_all.mat

% Uncomment features you wish NOT to include as predictors
% For example, to use only CAD-RADS as a predictor, uncomment every line 
% except for "data_table(:,'CADRADS')=[]".
%     data_table_all(:,(1:end-8))=[];%all vessel features
    data_table_all(:,'SPS')=[];
    data_table_all(:,'CADRADS')=[];
    data_table_all(:,'LeS')=[];
    data_table_all(:,'SIS')=[];
    data_table_all(:,'SSS')=[];
    
% Do you wish to run a completely independent test set after nested CV training?
    runindtest='no'; % 'yes' or 'no' 
    
    % If yes, do you want to make a new partition? 
    % In general, will want to use same partition to compare models
    makenewpart='yes'; % 'yes' or 'no'
    
    if strcmp(runindtest,'no') 
        % Use all the data
        data_table=data_table_all; 
        
    else
        % Stratify by all deaths
        if strcmp(makenewpart,'yes') 
            cvp_independent = cvpartition(table2array(data_table_all(:,{'outcome1_death'})), 'KFold', 3);
        elseif ~exist('cvp_independent','var')
            disp('error line 45: No partition exists yet; set makenewpart to yes (line 34)')
            return
        end
        
        % Divide data into 2/3 training and 1/3 test sets 
        data_table=data_table_all(cvp_independent.training(1),:);
        data_table_independent=data_table_all(cvp_independent.test(1),:); 
    end
    
% Designate number of Kfolds and repeats for inner and outer CV routines
  redundancy.numfolds_outer=5;   %5 
  redundancy.numfolds_inner=3;   %3
  redundancy.numrepeats_outer=30;%30
  redundancy.numrepeats_inner=1; %10 if hyperparmeters are being explored, otherwise 1
    
% Designate model algorithm
  for modelnumber=2
      
    if modelnumber==1
        modelname='Model1_LogisticRegression';
        
        %if using only a single predictor, save under its name instead of "LogR"
        if size(data_table_all,2)==4 
            savelabel=data_table_all.Properties.VariableNames{1};
        else
            savelabel='LogR';
        end
        
    elseif modelnumber==2
        modelname='Model2_KNearestNeighbors';
        savelabel='KNN';
    elseif modelnumber==3
        modelname='Model3_BaggedTrees';
        savelabel='BAG';
    elseif modelnumber==4
        modelname='Model4_ClassificationNeuralNet';
        savelabel='CNet';
    end
    
% Run nested crossvalidation for all deaths
    CCTAtable1=data_table;
    CCTAtable1.outcome2_CHDdeath=[];
    CCTAtable1.outcome3_CHDdeath_or_MI=[];
    [final_classifier1,ROC_outer1] = B2_nested_CV(CCTAtable1,modelname,redundancy,'all deaths');
    final_classifier1.runindtest=runindtest;
    disp(['all deaths: final best hyperparameters ' modelname])
    disp(final_classifier1.final_hp)
    
% Run nested crossvalidation for CHD deaths
    CCTAtable2=data_table;
    CCTAtable2.outcome1_death=[];
    CCTAtable2.outcome3_CHDdeath_or_MI=[];
    [final_classifier2,ROC_outer2] = B2_nested_CV(CCTAtable2,modelname,redundancy,'CHD deaths');
    final_classifier2.runindtest=runindtest;
    disp(['CHD deaths: final best hyperparameters ' modelname])
    disp(final_classifier2.final_hp)
    
% Run nested crossvalidation for CHD deaths + MI
    CCTAtable3=data_table;
    CCTAtable3.outcome1_death=[];
    CCTAtable3.outcome2_CHDdeath=[];
    [final_classifier3,ROC_outer3] = B2_nested_CV(CCTAtable3,modelname,redundancy,'CHD+MI');
    final_classifier3.runindtest=runindtest;
    disp(['CHD+MI: final best hyperparameters ' modelname])
    disp(final_classifier3.final_hp)

% Make results table
    summary_table=table(ROC_outer1.AUC_fold,ROC_outer2.AUC_fold,ROC_outer3.AUC_fold);
    summary_table.Properties.UserData.modelname=modelname;
    summary_table.Properties.VariableNames{'Var1'}=[savelabel '_AUC_alldeaths'];
    summary_table.Properties.VariableNames{'Var2'}=[savelabel '_AUC_CHDdeaths'];
    summary_table.Properties.VariableNames{'Var3'}=[savelabel '_AUC_CHDdeathsplusMI'];
    summary_table.Properties.UserData.CVmethod='AUC results from nested crossvalidation - fold level';
    disp(summary_table.Properties.UserData)
    disp(summary_table)

% Save
    if strcmp(runindtest,'yes')
        save(['results_independent/' savelabel '_summary_table'],'summary_table')
        save(['results_independent/' savelabel '_final_classifier1'],'final_classifier1');
        save(['results_independent/' savelabel '_ROC_outer1'],'ROC_outer1');
        save(['results_independent/' savelabel '_final_classifier2'],'final_classifier2');
        save(['results_independent/' savelabel '_ROC_outer2'],'ROC_outer2');
        save(['results_independent/' savelabel '_final_classifier3'],'final_classifier3');
        save(['results_independent/' savelabel '_ROC_outer3'],'ROC_outer3');
    else
        save(['results/' savelabel '_summary_table'],'summary_table')
        save(['results/' savelabel '_final_classifier1'],'final_classifier1');
        save(['results/' savelabel '_ROC_outer1'],'ROC_outer1');
        save(['results/' savelabel '_final_classifier2'],'final_classifier2');
        save(['results/' savelabel '_ROC_outer2'],'ROC_outer2');
        save(['results/' savelabel '_final_classifier3'],'final_classifier3');
        save(['results/' savelabel '_ROC_outer3'],'ROC_outer3');
    end
    
% Apply classifiers to independent test data
    if strcmp(runindtest,'yes')
         D1_call_predict_newdata
    end
    
  end 
  