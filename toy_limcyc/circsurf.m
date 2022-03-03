function [T, X, dotX] = circsurf(len, x0, tmax)

    assert(numel(x0)==2);
    x0 = reshape(x0, 1, 2);

    fun = @(t,x)f(t, x);
    [T, X] = ode45(fun, linspace(0, tmax, len), x0, []);

    dotX = zeros(size(X));
    for i=1:size(X,1)
        dotX(i,:) = f(T(i), X(i,:).').';
    end

end

function dxdt = f(t, x)
    tmp = x(1)^2+x(2)^2;
    dxdt = [x(1)-x(2)-x(1)*tmp; x(1)+x(2)-x(2)*tmp];
end