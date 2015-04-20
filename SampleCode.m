%% Sample Code

%% Normalisation
data_norm = bsxfun(@rdivide, data, sqrt(sum(data.^2,2)));

%% LPBoost;
data_dup = [data_norm, -data_norm]

[M, ~] = size(data_dup);
nu = 0.1:0.1:0.9;

group = zeros(M, length(nu));
acc = zeros(length(nu), 1);

for i = 1 : M
    data_new = data_dup;
    data_new(i, :) = [];
    
    label_new = label;
    label_new(i) = [];
    
    for j = 1 : length(nu)
        model = LPBoostYS(data_new,label_new,1/(nu(j)*(M-1)),16000);
        
        group(i, j) = sign(data_dup(i, model.idx) * model.a');
    end;
end;

for i = 1 : length(nu)
    acc(i) = sum(group(:, i) == label)/M;
end;

%% Dynamic LPBoost (Fixed Length)

% Set length
l = 5;

M = size(data_norm, 1);
miu = 0.1:0.1:0.9;

group = zeros(M, length(miu));
acc = zeros(length(miu), 1);

for i = 1 : length(miu)
    for j = 1 : M

        label_new = label;
        label_new(j) = [];
        data_new = data_norm;
        data_new(j, :) = [];
        
        model = dynamicLPBoostUnnorm(data_new,label_new,1/((M-1)*miu(i)),l,16000);
        
        data_new = zeros(1, size(model.sub, 2));
        for counter = 1 : size(model.sub, 2)
            data_new(counter) = featSelectUnnorm(data_norm(j, :), model.sub(:, counter),...
                model.atr{counter}, length(model.sub(:, counter)));
        end;
        group(j, i) = sign(data_new * model.a');
        
    end;
end;

for i = 1 : length(miu)       
    acc(i) = sum(group(:, i) == label)/M;
end;
