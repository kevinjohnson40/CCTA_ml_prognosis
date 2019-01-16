
%A_multiK_feature_selection.m
% Created 07-Apr-2018 Kevin M. Johnson, M.D. Yale University
% Revised 28-Aug-2018

% Uses relieff.m to assess importance of features for machine learning applied to
% coronary plaque features - goal is to find features with most prognostic value

%[RANKED,WEIGHT] = relieff(X,Y,K) computes ranks and weights of
%     attributes (predictors) for feature data matrix X and response 
%     vector Y using ReliefF algorithm for classification with K nearest
%     neighbors.

%Three outcomes were defined: all deaths, coronary artery disease deaths,
%and CHD deaths plus myocardial infarctions. The numbering of variables from 1
%through 3 generallly refers to these three outcomes.

%load data from a 6892 x 72 table: CCTAdataAugust2018.mat
%the first 64 columns are coronary artery segment features
%columns 65 through 69 are conventional scores (CAD-RADS, LeS, SPS, SSS, and SIS)
%columns 70 through 72 are outcomes (alldeaths, CHD, CHD_or_MI)
    load CCTAdataAugust2018.mat
    data_table=CCTAdataAugust2018;
    
%get features (predictors)
%skip last eight columns (those are conventional scores and outcomes, not features)
    X=table2array(data_table(:,1:end-8));
    
%get feature names
    feature_names=data_table.Properties.VariableNames(1:end-8);
    
%get outcomes
    alldeaths=table2array(data_table(:,end-2));
    CHD=table2array(data_table(:,end-1)); %coronary artery disease deaths
    CHDMI=table2array(data_table(:,end)); %CHD + myocardial infarctions
    
%run relieff algorithm for each outcome
%compute weights over multiple K values, save highest weight for each feature
%this helps avoid undervaluing a feature inadvertently. See Robnik-Sikonja M, 
%Kononenko I. Theoretical and Empirical Analysis of ReliefF and 
%RReliefF. Machine Learning 2003;53:23-69.

%define vector for number of nearest neighbors K
    K=10:10:100;

%all deaths
    wb=waitbar(0,'Calculating for all deaths ...');
    Y=alldeaths;
    wtall=nans(size(data_table,2)-8,length(K));
    for h=1:length(K)
        [~,wts]=relieff(X,Y,K(h));
        wtall(:,h)=wts';
        wbl=h/(3*length(K));
        waitbar(wbl,wb)
    end
    wts_highest=nanmax(wtall,[],2);
    wts_highest(isnan(wts_highest))=0;
    [wts1,ranks]=sort(wts_highest,'descend');
    fnames1=feature_names(ranks)';
    
%CHD deaths
    wb=waitbar(wbl,wb,'Calculating for CHD deaths ...');
    Y=CHD;
    wtall=nans(size(data_table,2)-8,length(K));
    for h=1:length(K)
        [~,wts]=relieff(X,Y,K(h));
        wtall(:,h)=wts';    
        wbl=(h+length(K))/(3*length(K));
        waitbar(wbl,wb)
    end
    wts_highest=nanmax(wtall,[],2);
    wts_highest(isnan(wts_highest))=0;
    [wts2,ranks]=sort(wts_highest,'descend');
    fnames2=feature_names(ranks)';

%CHD deaths + myocardial infarctions
    wb=waitbar(wbl,wb,'Calculating for CHD deaths + MI ...');
    Y=CHDMI;
    wtall=nans(size(data_table,2)-8,length(K));
    for h=1:length(K)
        [~,wts]=relieff(X,Y,K(h));
        wtall(:,h)=wts';
        wbl=(h+2*length(K))/(3*length(K));
        waitbar(wbl,wb)
    end
    wts_highest=nanmax(wtall,[],2);
    wts_highest(isnan(wts_highest))=0;
    [wts3,ranks]=sort(wts_highest,'descend');
    fnames3=feature_names(ranks)';
    delete(wb)
    
%make results table
    relieff_table=table(fnames1,wts1,fnames2,wts2,fnames3,wts3);
    relieff_table.Properties.VariableNames{'fnames1'}='alldeaths';
    relieff_table.Properties.VariableNames{'fnames2'}='CHDdeaths';
    relieff_table.Properties.VariableNames{'fnames3'}='CHDdeathsplusMI';
    
%save results
    save('input_data/relieff_table','relieff_table')
    writetable(relieff_table,'input_data/relieff_table.csv')
    
%Now go to B_threshold_inputs to set threshold cutoff values for the vessel
%features, and prepare the outcomes for further use

