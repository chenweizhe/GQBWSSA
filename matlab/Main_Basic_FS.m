clc,clear
close all


name = 'xxxx.csv';

res = readtable(name);
res = table2array(res);

r=randperm(size(res,1));
train_num = floor(length(r)*0.7);
train_data=res(r(1:train_num),:);      
test_data=res(r(train_num+1:end),:);   

xtrain   = train_data(:,1:end-1); ytrain  = train_data(:,end);
xvalid   = test_data(:,1:end-1);  yvalid  = test_data(:,end); 


[xtrain_re , x_PS] = mapminmax(xtrain',0,1);        xtrain_re = xtrain_re';
xvalid_re          = mapminmax('apply',xvalid',x_PS); xvalid_re = xvalid_re';


Multi_Num = 5; 
Popsize = 20;
Max_iter = 50;
fobj=@(x) Fitness_RF(x,0,xtrain_re,xvalid_re,ytrain,yvalid);
dim = size(res,2)-1;
lb = -6*ones(1,dim); 
ub = 6*ones(1,dim);



% [GWO_ind,GWO_conve,GWO_pos] = GWO(Popsize,Max_iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid);

% [WOA_ind,WOA_conve,WOA_pos] = WOA(Popsize,Max_iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid);

% [DBO_ind,DBO_conve,DBO_pos] = DBO(Popsize,Max_iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid);


% [HHO_ind,HHO_conve,HHO_pos] = HHO(Popsize,Max_iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid);



[GQBWSSA_ind,GQBWSSA_conve,GQBWSSA_pos] = GQBWSSA(Popsize,Max_iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid);





% mean_GWO_ind = mean(GWO_ind,2);
% mean_WOA_ind = mean(WOA_ind,2);
% mean_DBO_ind = mean(DBO_ind,2);
% mean_HHO_ind = mean(HHO_ind,2);
mean_GQBWSSA_ind = mean(GQBWSSA_ind,2);

% best_GWO = [min(GWO_conve(:,end)) std(GWO_conve(:,end)) mean(GWO_conve(:,end))];

% best_WOA = [min(WOA_conve(:,end)) std(WOA_conve(:,end)) mean(WOA_conve(:,end))];

% best_DBO = [min(DBO_conve(:,end)) std(DBO_conve(:,end)) mean(DBO_conve(:,end))];

% best_HHO = [min(HHO_conve(:,end)) std(HHO_conve(:,end)) mean(HHO_conve(:,end))];

best_GQBWSSA = [min(GQBWSSA_conve(:,end)) std(GQBWSSA_conve(:,end)) mean(GQBWSSA_conve(:,end))];


% [num,ind] = min(GWO_conve(:,end));besGWO_pos = GWO_pos(ind,:);
% [num,ind] = min(WOA_conve(:,end));besWOA_pos = WOA_pos(ind,:);
% [num,ind] = min(DBO_conve(:,end));besDBO_pos = DBO_pos(ind,:);
% [num,ind] = min(HHO_conve(:,end));besBWO_pos = HHO_pos(ind,:);
[num,ind] = min(GQBWSSA_conve(:,end));besGQBWSSA_pos = GQBWSSA_pos(ind,:);

all = [1 mean_GQBWSSA_ind' best_GQBWSSA besGQBWSSA_pos];



writecell({name ,'平均准确率','平均特征选择率','平均运行时间','最佳适应度','标准差','平均适应度均值','最佳特征子集','GQBWSSA'}...
   ,'xxxx.xlsx','WriteMode','append');
writematrix(all,'xxxx.xlsx','WriteMode','append');

