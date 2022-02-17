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
t_end=300;
s0=rand(N^2,1)./4; % initialise s near 0

a = 1;
lam = 6;
beta = 3/(lam^2);
gam = 1.05 * beta;
tau = 10;
dt = 0.5;
w0 = 6.7;
%w0 = (-12 * gam * beta) / (pi * gam * beta);


% Calculating matrix W
W=zeros(N^2,N^2);
for i=1:N^2
    for j=1:N^2
        [xi,yi] = getPos(i,N); % get position in grid of neuron
        [xj,yj] = getPos(j,N);
        dist = toroidalDist(xi,yi,xj,yj,N); % ensure periodicity

        %W(i,j) = DoG(dist, 0, 0, sqrt(1/(2*gam)), sqrt(1/(2*beta)), 1, 1);
        W(i,j) = w0*(a*exp(-gam*abs(dist).^2) - exp(-beta * abs(dist).^2));
    end
end

% Solve ODE 
[t,s] = ode23(@RHS,0:t_end,s0,[],W,tau);

% --- PLOT AND ANIMATE GRID CELLS ----
figure(1)
clf
for i=1:length(t)
    imagesc(reshape(s(i,:), [N N]));
    drawnow
end

% ------- FUNCTIONS ----------


% ODE \tau \frac{ds_i}{dt} = -s_i + f\[sum_j W_{ij}s_j + B_i \]
function RHS = RHS(t,s,W,tau)        
    RHS = (-s + f(W*s + 1))/tau;
    % ADD B_i(t) in here
end

% Rectification nonlinearity function
function f = f(x)   
    f = max(0,x);       
end

% Map list of points onto grid in order to find distances between points
function [x_p, y_p] = getPos(x,N)
        x_p = mod(x,N);
        y_p = floor(x/N)+1;
        if (x_p==0) % On right edge
            x_p=N;
        end
        if (y_p==N+1) % On top edge
            y_p=N;
        end
end

% Difference of Gaussians function - can rewrite in the normal 
function DoG = DoG(x,mu_1,mu_2,sig_1,sig_2,amp_1,amp_2)
    gaus = @(x,mu,sig,amp) amp*exp(-(((x-mu).^2)/(2*sig.^2)));
    DoG = gaus(x, mu_1, sig_1, amp_1) - gaus(x, mu_2, sig_2, amp_2);
end


function dist = toroidalDist(xi,yi,xj,yj,N)
    A = abs(xi - xj);
    B = abs(yi - yj);

    if (A > N/2)
        A = N - A;
    end
    if (B > N/2)
        B = N - B;
    end
    dist = sqrt(A^2 +B^2);
end
