function [res1,res2,res3] = GWO(Popsize,maxgen,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid)



Multi_Convergence_curve = []; 
Multi_time = [];
Multi_ACC = [];
Multi_Precision = [];
Multi_Recall = [];
Multi_Specificity = [];

for ii = 1:Multi_Num
    
    tic
    
    
    Alpha_pos=zeros(1,dim);
    Alpha_score=inf;
    
    Beta_pos=zeros(1,dim);
    Beta_score=inf;
    
    Delta_pos=zeros(1,dim);
    Delta_score=inf;
    
   
    IterCurve = zeros(1,maxgen);
    
   
    fitness = zeros(1,Popsize);
    for i = 1:Popsize
        Positions(i,:) = double(rand(1,dim)>0.5); 
        fitness(i) = fobj(Positions(i,:));
    end
    
   
    [SortFitness,indexSort] = sort(fitness);
    
    Alpha_pos = Positions(indexSort(1),:);
    Alpha_score = SortFitness(1);
    Beta_pos = Positions(indexSort(2),:);
    Beta_score = SortFitness(2);
    Delta_pos = Positions(indexSort(3),:);
    Delta_score = SortFitness(3);
    gBest = Alpha_pos;
    gBestFitness = Alpha_score;
    
    
    for t = 1:maxgen
        
        a=2-t*((2)/maxgen);
        for i = 1:Popsize
            for j = 1:dim
                
                r1=rand(); 
                r2=rand(); 
                A1=2*a*r1-a; 
                C1=2*r2; 
                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); 
                X1=Alpha_pos(j)-A1*D_alpha; 
                
               
                r1=rand();
                r2=rand();
                A2=5*a*r1-a;
                C2=5*r2; 
                D_beta=abs(C2*Beta_pos(j)-Positions(i,j));  
                X2=Beta_pos(j)-A2*D_beta; 
                
                
                r1=rand();
                r2=rand();
                A3=2*a*r1-a; 
                C3=2*r2;
                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
                X3=Delta_pos(j)-A3*D_delta; 
                
                EPositions(i,j)=(X1+X2+X3)/3;
            end
           
            EPositions(i,:) = BoundaryCheck(EPositions(i,:),ub,lb,dim);
        end
       
        for i = 1:Popsize
            Positions(i,:) = 1./(1+exp(EPositions(i,:)))>rand(1,dim);
            fitness(i) = fobj(Positions(i,:));
           
            if  fitness(i)<Alpha_score
                Alpha_score= fitness(i);
                Alpha_pos=Positions(i,:);
            end
            if  fitness(i)>Alpha_score &&  fitness(i)<Beta_score
                Beta_score= fitness(i); 
                Beta_pos=Positions(i,:);
            end
            if  fitness(i)>Alpha_score &&  fitness(i)>Beta_score &&  fitness(i)<Delta_score
                Delta_score= fitness(i); 
                Delta_pos=Positions(i,:);
            end
        end
        if t == maxgen
          
           save('finalfitgwo.mat',"fitness");
       
        end
        gBest = Alpha_pos;
        gBestFitness = Alpha_score;
        IterCurve(t) = gBestFitness;
    end

    
    FitnessValue = Fitness_RF(Alpha_pos,1,xtrain_re,xvalid_re,ytrain,yvalid);
    ACC=FitnessValue(2); Specificity=FitnessValue(3); 
    
    Multi_ACC(ii) = ACC;
   
    Multi_Specificity(ii) = Specificity;
   
    Multi_Convergence_curve(ii,:) = IterCurve;
    
    Multi_time(ii) = toc;
   
    Multi_Alpha(ii,:) = Alpha_pos;

end

res1 = [Multi_ACC;Multi_Specificity;Multi_time];
res2 = Multi_Convergence_curve;
res3 = Multi_Alpha;





