function model = LPBoostYS(X, Y, D, iter)
% Input:
% X: data
% Y: label
% D: regularisation parameter, usually D = 1/mv
% iter: maximum iteration

% Output:
% model: structure variable of LPBoost model

% Improved code
% Original code by: Dimitrios Athanasakis (dathanasakis [at] gotbim [dot] com)
% Yuxin Sun: yuxin.sun [at] ucl [dot] ac [dot] uk

[M, ~] = size(X);

% Initialise
counter = 1;
a = zeros(1, M);
beta = 0;
u = ones(M, 1)/M;
[val, ind] = max((u.*Y)'*X);

while (counter <= iter) && (val >= beta+eps)        
    hypo(:, counter) = X(:, ind);
    
    [u, a, beta] = LPcvx(hypo, Y, D);
    
    model.beta(counter) = beta;
    model.idx(counter) = ind;
    
    fprintf('ITERATION: %d CRITIRION: %d BETA: %d\n',counter, val, beta);
    counter = counter + 1;
    [val, ind] = max((u.*Y)'*X);
end;

model.a = a;
model.u = u;
