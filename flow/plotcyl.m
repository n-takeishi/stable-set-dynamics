function f = plotcyl(data)

load('CCcool.mat');

data = data*6.25; % to adjust colors

f = figure;
datatmin = -5; datamax = 5;
data(data>datamax) = datamax;
data(data<datatmin) = datatmin;

imagesc(data);
colormap(CC);

set(gca,'XTick',[],'XTickLabel',{})
set(gca,'YTick',[],'YTickLabel',{});
set(gcf,'Position',[100 100 600 260])
axis equal;
hold on;

contour(data,[-5.5:.5:-.5 -.25 -.125],':k','LineWidth',0.5)
contour(data,[.125 .25 .5:.5:5.5],'-k','LineWidth',0.5)

theta = (1:100)/100'*2*pi;
x = 49+25*sin(theta); y = 99+25*cos(theta);
fill(x,y,[.3 .3 .3]);
plot(x,y,'k','LineWidth',1);

set(gcf,'PaperPositionMode','auto');

caxis([-10.5 10.5]);

end
