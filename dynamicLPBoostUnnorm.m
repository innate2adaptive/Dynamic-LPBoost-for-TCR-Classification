function model = dynamicLPBoostUnnorm(K,Y,D,l,iter)
% Extend 3 spectrum kernel of length 3 to longer length
% Selects fixed-length substrings with uniform transition probabilities
% Dynamic programming LPBoost

% Input:
% K: m*8000 3-spectrum kernel matrix
% Y: desired labels
% D: regularisation parameter
% l: target length
% iter: max iterations

% Output:
% model: stuctured variable of dynamic LPBoost model

% Yuxin Sun: yuxin.sun [at] ucl [dot] ac [dot] uk

[m, ~]=size(K);
u=ones(m,1)/m;
beta=0;
F=[];
opt = [];

hypo = dynamicPosNegUnnorm(K, u, Y, m, l);

F = [F, hypo.kern];
crit = hypo.val;

counter = 1 ;

while( (counter<=iter)&& (crit>= (beta+eps) ) )
    
    opt(:,counter) = hypo.kern;
    model.idx(counter)=hypo.idx;
    model.atr{counter} = hypo.atr;
    model.kern(:, counter) = hypo.kern;
    model.sub(:, counter) = hypo.sub';
    model.subrvs(:, counter) = hypo.subrvs';
          
    [u, a,beta]=LPcvx(opt,Y,D);
    model.beta(counter)=beta;

    fprintf('ITERATION: %d CRITERION: %d, BETA: %d \n',counter,crit,beta);
    counter = counter+1;

    hypo = dynamicPosNegUnnorm(K, u, Y, m, l);
    crit = hypo.val;
    F = [F, hypo.kern];

end

model.a=a;
model.u=u;
