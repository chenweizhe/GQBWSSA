function [res1,res2,res3]=GQBWSSA(N,iter,lb,ub,dim,fobj,Multi_Num,xtrain_re,xvalid_re,ytrain,yvalid)



Multi_Convergence_curve = [];
Multi_time = [];
disp("==============0000=======================");

for ii  = 1:Multi_Num
    tic

    
    [pos,fit]=Q_G_initialization(N,dim,ub,lb,fobj);


    disp("==============1111=======================");

GQBWSSA_curve = zeros(1,iter);

[sorted_salps_fitness,sorted_indexes]=sort(fit);

xposbest = pos(sorted_indexes(1),:);
fvalbest = sorted_salps_fitness(1);



for t = 1:iter

        
        [best, blocation] = min(fit);
        if t == 1
            xposbest = pos(blocation, :); 
            fvalbest = best; 
        elseif best < fvalbest
            fvalbest = best;
            xposbest = pos(blocation, :);
        end


        
        disp('=================');
        disp(t);
        
        

        c1 = 2*exp(-(4*t/iter)^2); % Eq. (3.2) in the paper

        for i=1:size(pos,1)

    %         pos= pos';

            if i<=N/2
                for j=1:1:dim
                    c2=rand();
                    c3=rand();
                   mine=0.4;
                maxe=0.9;
                w=maxe+t*(maxe-mine)/iter;
                    if c3<0.5 
                        pos(i,j)=w*xposbest(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                         pos(i, :) = max(pos(i, :), lb); pos(i, :) = min(pos(i, :), ub);   

                    else
                        pos(i,j)=w*xposbest(j)-c1*((ub(j)-lb(j))*c2+lb(j));

                        pos(i, :) = max(pos(i, :), lb); pos(i, :) = min(pos(i, :), ub);   
                    end

                end

            elseif i>N/2 && i<N+1
                for j=1:1:dim
                point1=pos(i-1,:);
                point2=pos(i,:);
                MMean=mean(pos(i,:));
                beta1=0.4;
                beta2=0.9;
                beta=beta2-t*(beta2-beta1)/iter;
                if rand>0.1 
                pos(i,:)=(point2+point1)/2; % % Eq. (3.4) in the paper
                pos(i, :) = max(pos(i, :), lb); pos(i, :) = min(pos(i, :), ub);   
                else

                pos(i,:)= xposbest-beta*(pos(i,:) - 2*MMean);
                pos(i, :) = max(pos(i, :), lb); pos(i, :) = min(pos(i, :), ub);       

                end
                end
            end

    %         pos= pos';
        end


     
       for i=1:N
         pos(i, :) = max(pos(i, :), lb); 
         pos(i, :) = min(pos(i, :), ub);   
         fit(i) = fobj(pos(i, :));

         [best, blocation] = min(fit);

        if best < fvalbest
            fvalbest = best;
            xposbest = pos(blocation, :);
        end
       end
       
        for i = 1:size(xposbest,2)
            if i ==1
                xposbest(i) =  xposbest(i);
            else
                xposbest(i) =  round(xposbest(i));
            end  
        end
        if t == iter
           save('finalfitssa.mat',"fit");
       end
        GQBWSSA_curve(t) = fvalbest;
end

    
    xposbest = xposbest>0;
    
    FitnessValue = Fitness_RF(xposbest,1,xtrain_re,xvalid_re,ytrain,yvalid);
    ACC=FitnessValue(2); Specificity=FitnessValue(3); 
   
    Multi_ACC(ii) = ACC;
   
    Multi_Specificity(ii) = Specificity;
    
    Multi_Convergence_curve(ii,:) = GQBWSSA_curve;
    
    Multi_time(ii) = toc;
    
    Multi_Alpha(ii,:) = xposbest;
    disp('&&&&&&&');

end
res1 = [Multi_ACC;Multi_Specificity;Multi_time];
res2 = Multi_Convergence_curve;
res3 = Multi_Alpha;


