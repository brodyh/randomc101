function [y,yw,dw,db] = logreg(x,target,w,b,istest)
%%
% Calculates a weighted linear sum of inputs and passes throgh cross
% entropy loss. Returns the function value and derivatives
%

if (nargin == 4)
    istest = 0;
end

%% function value calculation
yw = w*(x(:)) + b;
smden = sum(exp(yw));
sm = exp(yw)/smden;
y = (-target')*log(sm);

if istest == 0 % training
    %% derivative calculation
    % dy/d(yw)
    dy = sm -target;
    % dy/dw = dy/d(yw) * d(yw)/dw
    % d(yw) / dw = x
    dw = dy*x';
    % dy/db = dy/d(yw) * d(yw)/db = dy/d(dw)
    db = dy;
else % testing
    dw = 0;
    db = 0;
end

