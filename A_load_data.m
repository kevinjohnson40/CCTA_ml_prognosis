
%A_load_data

% CCTA data analysis
% Kevin M. Johnson, M.D. Yale University, March 2019
% This is the initial routine.

% Loads vessel features (predictors) and outcomes (responses) derived from 
% patient database. 64 columns represent vessel features followed by 
% 3 outcome columns.
    load input_data/CCTAdataMarch2019
    data_table_all=allinputsoutcomes;
    
% Shuffle rows because original data is in chronological order.
% This helps avoid possible bias.
    data_table_all = data_table_all(randperm(size(data_table_all,1)),:);
    
% Experiment to randomize outcomes - can be used to check for bias in modeling
%     newend = data_table_all(randperm(size(data_table_all,1)),end-2:end);
%     data_table_all(:,end-2:end)=newend;
    
% Add interaction inputs here if desired
    
% Eliminate predictors with zero variance
    predictors=table2array(data_table_all(:,1:end-1));
    temp=nanstd(predictors);
    colpos=find(temp==0);
    data_table_all(:,colpos)=[];

% Save
    save('input_data/data_table_all','data_table_all')
    
%next Step: go to B1_call_nested_CV.m
