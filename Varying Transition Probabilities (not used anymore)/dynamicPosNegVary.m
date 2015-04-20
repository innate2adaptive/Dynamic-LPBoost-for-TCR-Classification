function hypo = dynamicPosNegVary(K, u, Y, m, l)

% Select max kernel on both postive and negative instances

hypo(1) = dynamicPVary(K, u, Y, m, l); % positive
hypo(2) = dynamicPVary(-K, u, Y, m, l); % negative

[~, hypoidx] = max([hypo.val]);
hypo = hypo(hypoidx);

% Select max hypo between positive and negative choice
if hypoidx == 1  % positive
    
    hypo.atr = 'Positive';
    l = length(hypo.sub);
    kern = featSelectVary(K, hypo.sub, hypo.atr, l);
    
    % Convert linear subscripts into index
    temp = num2cell(hypo.subrvs);
    hypo.idx = sub2ind(20*ones(1, l), temp{:});
    
else  % negative

    hypo.atr = 'Negative';    
    l = length(hypo.sub);
    kern = featSelectVary(K, hypo.sub, hypo.atr, l);
    
    % Convert linear subscripts into index
    temp = num2cell(hypo.subrvs);
    hypo.idx = sub2ind(20*ones(1, l), temp{:})+20^l;
end;

hypo.kern = kern;  % column that gives out the max
