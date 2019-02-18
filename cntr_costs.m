clc;
% clear;

f1 = figure(1);
clf( f1 );
ax1 = axes();
ax1.NextPlot = 'add';

load opt_costs_grid.mat

[kd_mesh, kp_mesh] = meshgrid( kd_array, kp_array );

num_contours = 6;

contr{1,1}  = costs;
contr{2,1}  = rt_ratio_grid;
contr{3,1}  = zeta_lon_grid;
contr{4,1}  = zeta_lat_grid;
contr{5,1}  = zeta_th_grid;
contr{6,1}  = mot_grid;


cm = cell( num_contours, 1 );
co = cell( num_contours, 1 );

co_z            = contr;
co_line_clr     = { 'k'; 'g'; 'b'; 'm'; 'r'; 'c' };
co_line_sty     = { '-'; '-'; '-'; '-'; '-'; '-' };
co_levels(2:num_contours,1)  = { [8, 8], [0.607, 0.807], [0.607, 0.807], ...
                                [0.607, 0.807], [15, 15]};
%

co_levels{1,1}  = [ 39, 40, 43.622, 50 ,3 .^ [1:5]];

% init figure(s)
f1 = figure(1);
clf( f1 );
ax1 = axes();
ax1.NextPlot = 'add';

% create contours with properties
for ii = 1 : num_contours
    [cm{ii,1}, co{ii,1}]    = contour( kd_mesh, kp_mesh, co_z{ii,1} );
    co{ii,1}.LevelList      = co_levels{ii,1};
    co{ii,1}.LineColor      = co_line_clr{ii,1};
    co{ii,1}.LineStyle      = co_line_sty{ii,1};

    if ii == 1 || ii == 3 || ii == 4 || ii == 5
        co{ii}.ShowText     = 'on';
        co{ii}.LabelSpacing = 250;
    end
end

% create Optimal point
pl1 = plot( -0.0807, -0.0509  );

pl1.LineStyle       = 'none';
pl1.Marker          = 'o';
pl1.MarkerEdgeColor = 'r';
pl1.MarkerSize      = 15;
pl1.LineWidth       = 3;


% show a labels and legend
ti1 = title( 'PID Tuner Contour' );
xl1 = xlabel( 'kp_z' );
yl1 = ylabel( 'kd_z' );
leg = legend(   'Cost', ...
                't_r ratio', ...
                '\zeta_h', ...
                '\zeta_z', ...
                '\zeta_{\theta}', ...
                'max fl,fr', ...
                'Optimal Design Pt for Cost' );
%
leg.Location = 'best';

%
