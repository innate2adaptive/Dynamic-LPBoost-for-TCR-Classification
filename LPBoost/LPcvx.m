function [u, a, betaa] = LPcvx(X, Y, D)
% Input:
% X: hypothesis space up to date
% Y: label
% D: regularisation parameter

% Output:
% u: misclassification cost
% a: coefficients
% betaa: objective

[M, ~] = size(X);
Z = bsxfun(@times, X, Y);

cvx_begin quiet
    variable betaa
    variable u(M, 1)
    dual variable a
    dual variable xi
    
    minimize betaa
    subject to
    
    a: u'*Z <= betaa
    u >= 0
    sum(u) == 1;
    xi: u <= D
cvx_end

a(a<=eps)=0;