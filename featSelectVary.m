function kern = featSelectVary(K, sub, atr, l)
% Generate data set with selected features

[m, ~] = size(K);
K_reshape = reshape(K, m, 20, 20, 20);
kern = zeros(m, 1);

for i = 1 : (l-3+1)
    temp = K_reshape(:, sub(i+2), sub(i+1), sub(i));
    kern = kern + temp(:);
end;

nml = sqrt(l-3+1);

kern = kern/nml;  % normalisation

if strcmp(atr, 'Negative')
    kern = -kern;
end;