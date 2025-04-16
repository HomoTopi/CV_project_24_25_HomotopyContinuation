% decomposeDegenerateConic -  decompose in two lines the given degenerate
% conic (including complex decompositions)
%              - Pierluigi Taddei (pierluigi.taddei@polimi.it)
%
% Usage:     [l m] = decomposeDegenerateConic(C)
%
% Arguments:
%           c - degenerate symmetric 3x3 matrix of the conic
%           l m - homogeneous line coordinates
% 07.3.2007 : Created
%
%
function [l m] = decomposeDegenerateConic(c)

if (rank(c) == 1) %c is rank 1: direct split is possible
    C = c;
else %rank 2: need to detect the correct rank 1 matrix
    %% use the dual conic of c
    B = -adjointSym3(c);

    %% detect intersection point p
    [maxV di] = max(abs(diag(B)));
    i = di(1);
    
    if (B(i,i) <0)
        l = [];
        m = [];
        return;
    end
    b = sqrt(B(i,i));  % This will be complex when B(i,i) < 0
    p = B(:,i)/b;

    %% detect lines product
    Mp = crossMatrix(p);

    C = c + Mp;
end
%% recover lines
[maxV ci] = max(abs(C(:)));
j = floor((ci(1)-1) / 3)+1;
i = ci(1) - (j-1)*3;

l = C(i,:)';
m = C(:,j);

end