function FitnessValue=Fitness_RF(x,tt,xtrain_re,xvalid_re,ytrain,yvalid)

if tt==1
    flag=1;
else
    flag=0;
end

alpha = 0.99;
beta = 0.01;

x=x>0.5;
x=cat(2,x,zeros(size(x,1),1));
x=logical(x);


if sum(x)==0
    if flag==1
        FitnessValue=ones(1,3)*inf;
        return;
    else
        FitnessValue=ones(1)*inf;
        return;
    end
end

xtrain_re = xtrain_re(:,x);
xvalid_re = xvalid_re(:,x);


ntree = 20;
mtry  = 2; 

model = classRF_train(xtrain_re,ytrain,ntree,mtry);

pred = classRF_predict(xvalid_re,model);


% Accuracy
ACC      = (sum(pred == yvalid) / length(yvalid));
fit=alpha*(1-ACC)+beta*sum(x)/(length(x)); %fitness function
% fit
if flag==1
    FitnessValue=[fit, ACC,sum(x)/(length(x))]; 
else
    FitnessValue = fit;
end
