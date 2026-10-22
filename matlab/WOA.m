function [res1,res2,res3] = WOA(Popsize,maxgen,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid)



lb= lb.*ones( 1,dim );
ub= ub.*ones( 1,dim );

Multi_Convergence_curve = []; 
Multi_time = [];
Multi_ACC = [];
Multi_Specificity = [];
for ii = 1:Multi_Num
    
    tic

   
    fitness = zeros(1,Popsize);
    Positions = zeros(Popsize,dim); 
    for i = 1:Popsize
        Positions(i,:) = double(rand(1,dim)>0.5) ;
        fitness(i) = fobj(Positions(i,:));
    end
    [SortFitness,indexSort] = sort(fitness); 
    
    Leader_pos = Positions(indexSort(1),:);
    Leader_score = SortFitness(1);

    

    
    Convergence_curve=zeros(1,maxgen);
    

    for t=1:maxgen
        
        
        a=2-t*((2)/maxgen);
        
        a2=-1+t*((-1)/maxgen);
        
       
        for i=1:size(Positions,1)
            
            r1=rand();
            r2=rand();
            A=2*a*r1-a;
            C=2*r2;
          
            b=1;
            l=(a2-1)*rand+1;
           
            for j=1:size(Positions,2)
                
                p = rand();
                if p<0.5
                    if abs(A)>=1
                        
                        rand_leader_index = randi([1 Popsize]);
                        X_rand = Positions(rand_leader_index, :);
                        D_X_rand=abs(C*X_rand(j)-Positions(i,j));
                        EPositions(i,j)=X_rand(j)-A*D_X_rand;
                    elseif abs(A)<1
                        
                        D_Leader=abs(C*Leader_pos(j)-Positions(i,j));
                        EPositions(i,j)=Leader_pos(j)-A*D_Leader;
                    end
                elseif p>=0.5
                   
                    distance2Leader=abs(Leader_pos(j)-Positions(i,j));
                    EPositions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                end
            end
        end
        
        
        for i=1:size(Positions,1)
            Flag4ub=EPositions(i,:)>ub;
            Flag4lb=EPositions(i,:)<lb;
            EPositions(i,:)=(EPositions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            Positions(i,:) = 1./(1+exp(EPositions(i,:)))>rand(1,dim);
            fitness(i)=fobj(Positions(i,:));
            
            if fitness(i)<Leader_score
                Leader_score=fitness(i);
                Leader_pos=Positions(i,:);
            end
        end
        
        Convergence_curve(t)=Leader_score;
        if t == maxgen
          
           save('finalfitwoa.mat',"fitness");
       
        end
    end
    

    FitnessValue = Fitness_RF(Leader_pos,1,xtrain_re,xvalid_re,ytrain,yvalid);
    ACC=FitnessValue(2); Specificity=FitnessValue(3); 
    
    Multi_ACC(ii) = ACC;
   
    Multi_Specificity(ii) = Specificity;
    
    Multi_Convergence_curve(ii,:) = Convergence_curve;
   
    Multi_time(ii) = toc;
   
    Multi_Alpha(ii,:) = Leader_pos;

end

res1 = [Multi_ACC;Multi_Specificity;Multi_time];
res2 = Multi_Convergence_curve;
res3 = Multi_Alpha;








