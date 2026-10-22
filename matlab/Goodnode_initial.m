function X = Goodnode_initial(pop,ub,lb,dim)

[GD] = Goodnode(pop,dim);
X = zeros(pop,dim); 
for i = 1:pop
    for j = 1:dim
        X(i,j) = (ub(j) - lb(j))*GD(i,j) + lb(j);  
    end
        X(  i , : ) = 1./(1+exp(X(  i , : )))>rand(1,dim);
end
end


function [GD] = Goodnode(M,N)
% M is the number of points; N is the dimension
if (nargin==0)
    M=1000;
    N=2;
end
%%
tmp1 = [1: M]'*ones(1, N);
Ind = [1: N];
prime1 = primes(100*N);
[p,q]=find(prime1 >= (2*N+3));
tmp2 = (2*pi.*Ind)/prime1(1,q(1));
tmp2 = 2*cos(tmp2);
tmp2 = ones(M,1)*tmp2;
GD = tmp1.*tmp2;
GD = mod(GD,1);

end