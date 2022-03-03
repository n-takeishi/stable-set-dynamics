function [T, X, dotX] = vdp(len, x0, mu, tmax)

    if nargin<4, tmax=20; end

    assert(numel(x0)==2);
    x0 = reshape(x0, 1, 2);

    fun = @(t,x)f(t, x, mu);
    [T, X] = ode45(fun, linspace(0, tmax, len), x0, []);

    dotX = zeros(size(X));
    for i=1:size(X,1)
        dotX(i,:) = f(T(i), X(i,:).', mu).';
    end

end

function dxdt = f(t, x, mu)
    dxdt = [x(2); mu*(1-x(1)^2)*x(2)-x(1)];
end