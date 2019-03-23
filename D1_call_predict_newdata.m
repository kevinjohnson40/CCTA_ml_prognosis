
% Script - D1_call_predict_newdata

% Called from B1_call_nested_CV.m after nested cross-validation training 

% CCTA data analysis
% Kevin M. Johnson, M.D. Yale University, March 2019

% Three outcomes are defined: all deaths, coronary artery disease deaths,
% and CHD deaths plus myocardial infarctions. Variable numbering 1
% through 3 generally refers to these three outcomes. 
    
% All deaths
    CCTAtable1_independent=data_table_independent;
    CCTAtable1_independent.outcome2_CHDdeath=[];
    CCTAtable1_independent.outcome3_CHDdeath_or_MI=[];
    ROC_independent1=D2_predict_newdata(CCTAtable1_independent,final_classifier1,redundancy,'all deaths');
    
% CHD deaths
    CCTAtable2_independent=data_table_independent;
    CCTAtable2_independent.outcome1_death=[];
    CCTAtable2_independent.outcome3_CHDdeath_or_MI=[];
    ROC_independent2=D2_predict_newdata(CCTAtable2_independent,final_classifier2,redundancy,'CHD deaths');

% CHD deaths + MI
    CCTAtable3_independent=data_table_independent;
    CCTAtable3_independent.outcome1_death=[];
    CCTAtable3_independent.outcome2_CHDdeath=[];
    ROC_independent3=D2_predict_newdata(CCTAtable3_independent,final_classifier3,redundancy,'CHD+MI');
    
% Make new results table
    summary_table_independent=table(ROC_independent1.AUC,ROC_independent2.AUC,ROC_independent3.AUC);
    summary_table_independent.Properties.UserData.modelname=modelname;
    summary_table_independent.Properties.VariableNames{'Var1'}=['test_' savelabel '_AUC_alldeaths'];
    summary_table_independent.Properties.VariableNames{'Var2'}=['test_' savelabel '_AUC_CHDdeaths'];
    summary_table_independent.Properties.VariableNames{'Var3'}=['test_' savelabel '_AUC_CHDdeathsplusMI'];
    summary_table_independent.Properties.UserData.comment='AUC results on independent test data';
    disp(summary_table_independent.Properties.UserData)
    disp(summary_table_independent)

% Save
    save(['results_independent/' savelabel '_summary_table_independent'],'summary_table_independent')
    save(['results_independent/' savelabel '_ROC_independent1'],'ROC_independent1');
    save(['results_independent/' savelabel '_ROC_independent2'],'ROC_independent2');
    save(['results_independent/' savelabel '_ROC_independent3'],'ROC_independent3');
    