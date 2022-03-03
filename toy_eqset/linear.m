function [T, X, dotX] = linear(len, x0, tmax, a, gamma)
    assert(gamma>0);

    assert(numel(x0)==2);
    x0 = reshape(x0, 1, 2);

    fun = @(t,x)f(t, x, a, gamma);
    [T, X] = ode45(fun, linspace(0, tmax, len), x0, []);

    dotX = zeros(size(X));
    for i=1:size(X,1)
        dotX(i,:) = f(T(i), X(i,:).', a, gamma).';
    end

end

function dxdt = f(t, x, a, gamma)
    dxdt = [-(x(2)-a)*x(1); gamma*x(1)^2];
end
