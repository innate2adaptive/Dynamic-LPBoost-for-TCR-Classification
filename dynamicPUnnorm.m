function model = dynamicPUnnorm(K, u, Y, m, l)
% Dynamic programming to extend the 3-spectrum kernel to lengths longer
% than 3
% Normalisation: K'(s, t) = K(s, t)/ sqrt(K(s, s)*K(t, t))
% Precompute: K(s, t)/ sqrt(K(s, s)) as K
% K(t, t) is computed here as length(t)-spectrum+1

% Yuxin Sun: yuxin.sun [at] ucl [dot] ac [dot] uk

% Reshape K to a multi-dimensional matrix
K_reshape = reshape(K, m, 20, 20, 20);

% Initialise subscripts
ind = zeros(1,l);

% Initialise size 400 table
for i = 2 : l
    tab(i-1).val = zeros(20, 20); % size 400 table of kernel values
    tab(i-1).sub = zeros(20, 20); % size 400 table of subscripts
end;

% Dynamic programming to select features in LPBoost
for len = (3-1) : (l-1)
    for ltr2  = 1 : 20
        for ltr1 = 1 : 20
            temp = zeros(1, 20);
            nml = zeros(1, 20);
            for ltr3 = 1 : 20
                temp(ltr3) = tab(len-1).val(ltr3, ltr2)...
                    + sum((u.*Y)'*K_reshape(:, ltr1, ltr2, ltr3))/sqrt(l-3+1); 
            end;
            
            [max_val, max_sub] = max(temp);
            tab(len).val(ltr2, ltr1) = max_val;
            tab(len).sub(ltr2, ltr1) = max_sub;
            
        end;
    end;
end;

% Track subscripts
temp = tab(l-1).val;
[~, temp] = max(temp(:));
[ind(l-1), ind(l)] = ind2sub(size(tab(l-1).val), temp);

for i = l-2 : -1 : 1
    ind(i) = tab(i+1).sub(ind(i+1), ind(i+2));
end;


model.val = max(max(tab(l-1).val));
model.sub = ind;
model.subrvs = fliplr(ind);