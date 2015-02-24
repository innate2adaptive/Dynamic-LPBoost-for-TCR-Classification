function model = dynamicPVary(K, u, Y, m, l)
% Dynamic programming to extend the 3-spectrum kernel to lengths longer
% than 3

% Reshape K to a multi-dimensional matrix
K_reshape = reshape(K, m, 20, 20, 20);

% Initialise subscripts
ind = zeros(1,l);
feat = zeros(1,l);
feat_ind = zeros(1,l);

% Initialise size 400 table
for i = 2 : l
    tab(i-1).val = zeros(20, 20); % size 400 table of kernel values
    tab(i-1).sub = zeros(20, 20); % size 400 table of subscripts
end;

for len = (3-1) : (l-1)
    for ltr2  = 1 : 20
        for ltr1 = 1 : 20
            temp = zeros(1, 20);
            for ltr3 = 1 : 20
                temp(ltr3) = tab(len-1).val(ltr3, ltr2)...
                    + sum((u.*Y)'*K_reshape(:, ltr1, ltr2, ltr3));  % /nml: normalise
            end;
            
            [max_val, max_sub] = max(temp);
            tab(len).val(ltr2, ltr1) = max_val;
            tab(len).sub(ltr2, ltr1) = max_sub;
        end;
    end;
    
    [feat(len+1), feat_ind(len+1)] = max(tab(len).val(:)/sqrt(len+1-3+1));  % feat: max kernel value at current iteration; feat_ind: index of feat on most recent table
    
end;

[max_feat, ind_feat] = max(feat);
temp = feat_ind(ind_feat);
% Track subscripts
% temp = tab(ind_feat-1).val;
% [~, temp] = max(temp(:));

[ind(ind_feat-1), ind(ind_feat)] = ind2sub(size(tab(ind_feat-1).val), temp);

for i = ind_feat-2 : -1 : 1
    ind(i) = tab(i+1).sub(ind(i+1), ind(i+2));
end;

ind(ind_feat+1:end) = [];

model.val = max_feat;
model.sub = ind;
model.subrvs = fliplr(ind);