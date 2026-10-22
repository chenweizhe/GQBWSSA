function [x,fit]=Q_G_initialization(pop,dim,ub,lb,fobj)


x = Goodnode_initial(pop,ub,lb,dim);

fit=ones(1,pop);
x2=x;
fit2=fit;
byx1=x;
byx2=x;

for i = 1 : pop
    
    fit(i) = fobj(x(i,:)) ;  
    theta=asin(2*(x(i,:)-lb)./(ub-lb)-1);  
    x2(i,:)=0.5*(1+cos(theta)).*(ub-lb)+lb;
    x2(i,:) = Bounds(x2(i,:), lb, ub);
    x2(  i , : ) = 1./(1+exp(x2(  i , : )))>rand(1,dim);
    fit2(i)=fobj(x2(  i , : ));
    if fit2(i)<fit(i)   
        x(i,:)=x2(i,:);
        fit(i)=fit2(i);
    end
end
disp("==============2222=======================");

function s = Bounds(s, Lb, Ub)
  temp = s;
  I = temp < Lb;     
  temp(I) = Lb(I);   
  J = temp > Ub;       
  temp(J) = Ub(J);  
  s = temp; 
end

end