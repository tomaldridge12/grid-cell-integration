%% Accurate Path Integration in Continuous Attractor Network Models of Grid Cells, Y. Burak & I. Fiete
% VARIABLES
% tau : time constant
% s : neural activity
% dt : time step
% N : number of neurons in network
% e_\theta_j : unit vector pointing in direction \theta
% W_ij : synaptic weight from neuron j to neuron i
% B : sensory input to neuron i
%
% MODEL
% ds_i/dt = -s_i + ReLu([Ws]i + B_i)

% --- INITIALSE VARS ---

N = 32;
t_end=500;
s0=rand(N^2,1)./4; % initialise s near 0 - starting with hexagonal grid would improve the convergence rate

% Use initial hexagon data if N = 32 to get better convergence
if (N == 32)
    load('Hexagons32.mat', 'u');
    s0 = u;
end

a = 1;
lam = 6;
beta = 3/(lam^2);
gam = 1.05 * beta;
tau = 10;
dt = 0.5;
w0 = 6.7;
%w0 = (-12 * gam * beta) / (pi * gam * beta);
l = 1;
alpha = 0.10315;


% Calculate directional preference [E, N; W, S]
% E = [1; 0]
% N = [0; 1]
% W = [-1; 0]
% S = [0; -1]
x_pref = repmat([-1, 1; 0, 0], N/2); % x component of directional preference, tiled to form a grid
y_pref = repmat([0, 0; -1, 1], N/2); % y component of directional preference, tiled to for, a grid
% Reshape matrix of preferences to vector
x_prefCol = reshape(x_pref,N^2,1);
y_prefCol = reshape(y_pref,N^2,1);
% Combine preference vectors to e matrix
e = [x_prefCol, y_prefCol];

% Calculating matrix W
W=zeros(N^2,N^2);
for i=1:N^2
    for j=1:N^2
        % Get position of neurons i,j in grid using function getPos 
        [xi,yi] = getPos(i,N);
        [xj,yj] = getPos(j,N);

        %update xj, yj with directional preference
        xj1 = xj + l*x_pref(xj, yj);
        yj1 = yj + l*y_pref(xj, yj);
        xj = xj1;
        yj = yj1;

        % Apply distance over a torus to ensure periodic conditions
        dist = toroidalDist(xi,yi,xj,yj,N);
        % Pass distance into difference of gaussians function
        W(i,j) = w0*(a*exp(-gam*abs(dist).^2) - exp(-beta * abs(dist).^2));
    end
end

% % Vector of velocities - currently goes in the wrong x direction? Very weird
% v = [0.5; 0.3];
% v = repmat(v,1, t_end/2);
% v2 = [-0.5; -0.5];
% v2 = repmat(v2, 1, t_end/2);
% v = [v, v2];

% Load positions from rat movement file
load('Hafting_Fig2c_Trial2.mat', 'pos_x')
load('Hafting_Fig2c_Trial2.mat', 'pos_y')
t_v = 1:t_end;
% Initialise vectors
v_x = zeros(length(t_v),1) ;
v_y = zeros(length(t_v),1) ;
% Calculate velocities from position vectors
for i = 1:length(t_v)-1
    v_x(i) = (pos_x(i+1)-pos_x(i))/(t_v(i+1)-t_v(i)) ;
    v_y(i) = (pos_y(i+1)-pos_y(i))/(t_v(i+1)-t_v(i)) ;
end
v = 2*[v_x, v_y]';

% Solve ODE 
[t,s] = ode23(@RHS,0:t_end,s0,[],W,tau,alpha,e,v);

% --- PLOT AND ANIMATE GRID CELLS ----
figure(1)
clf
for i=1:length(t)
    imagesc(reshape(s(i,:), [N N]));
    drawnow
end

% ------- FUNCTIONS ----------

% ODE \tau \frac{ds_i}{dt} = -s_i + f\[sum_j W_{ij}s_j + B_i \]
function RHS = RHS(t,s,W,tau,alpha,e,v)
    % A = ones(size(s,2),1);
    
    % ODE uses adaptive time stepping. We normalise these time steps so we
    % can use our velocity vectors at specific time steps. Since t is
    % continuous on [0, t_end] we take the floor and +1 to avoid using zero
    % as an index. As we plus 1, this can take us outside the range of the
    % vector. We check this edge case and reassign if neccessary. 
    velT = floor(t)+1;
    if velT == size(v,2)+1
        velT=size(v,2);
    end

    % B = A * (1 + alpha * direction vectors * velocity)
    B = 1*(1 + alpha*(e*v(:,velT)));

    % Main ODE
    RHS = (-s + f(W*s + B))/tau;
end

% Rectification nonlinearity function
function f = f(x)   
    f = max(0,x);       
end

% Map list of points onto grid in order to find distances between points
function [x_p, y_p] = getPos(x,N)
        x_p = mod(x,N);
        if (x==N)
            x_p=N;
            y_p=floor(x/N);
        else
            y_p = floor(x/N + 1);
        end
        if (x_p==0) % On right edge
            x_p=N;
        end
        if (y_p==N+1) % On top edge
            y_p=N;
        end
end

% Difference of Gaussians function - not currently used.
function DoG = DoG(x,mu_1,mu_2,sig_1,sig_2,amp_1,amp_2)
    gaus = @(x,mu,sig,amp) amp*exp(-(((x-mu).^2)/(2*sig.^2)));
    DoG = gaus(x, mu_1, sig_1, amp_1) - gaus(x, mu_2, sig_2, amp_2);
end


function dist = toroidalDist(xi,yi,xj,yj,N)
    A = abs(xi - xj);
    B = abs(yi - yj);
    
    % If A or B is over half the length of N, then we know we can use
    % periodic boundary conditions to find a closer value.
    if (A > N/2)
        A = N - A;
    end
    if (B > N/2)
        B = N - B;
    end
    dist = sqrt(A^2 +B^2);
end
