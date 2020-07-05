function y = nnSoftThresholding(x, lambda)
% SHRINKAGE performs the shrinkage operation (soft- thresholding), with 
% parameter lambda, on the input signal x.
%
% It is also the evaluation of the proximity operator of g(z) = 
% lambda * norm(z,1) at point x.
%
%   Usage:  y = shrinkage(x, lambda)
%
%   Input parameters:
%       x         : Input signal
%       lambda    : Soft-thresholding parameter
%         
%   Output parameters:
%       y         : Shrinkage (soft-thresholding) of x
%
%   Example:
%       z = linspace(-1,1,100);
%       x = z.^3;
%       lambda = 0.5;
%       y = shrinkage(x, lambda);
%       plot(z,x,'-b'); hold on; plot(z,y,'-r');
%       legend('Original signal', 'Shrinked signal');
%
%   See also: FISTA.m
%       
%   References:
%
% Author: Rodrigo Pena (rodrigo.pena@epfl.ch)
% Date: 26 Oct 2015
% Testing:

% y = max( abs(x) - lambda, zeros(size(x)) ).* sign(x); % Usual
y = max( x - lambda, zeros(size(x)) ); % Non-negative


end   