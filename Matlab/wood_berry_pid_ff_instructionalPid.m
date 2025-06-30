% Define s-variable
s = tf('s');

% Wood & Berry MIMO Transfer Function Matrix
G = [1.0*exp(-10*s)/(11*s+1), 0.2*exp(-7*s)/(8*s+1), 0.3*exp(-3*s)/(10*s+1);
     0.1*exp(-5*s)/(9*s+1), 1.2*exp(-6*s)/(12*s+1), 0.2*exp(-8*s)/(7*s+1);
     0.3*exp(-4*s)/(13*s+1), 0.4*exp(-2*s)/(6*s+1), 1.1*exp(-9*s)/(5*s+1)];

% Design simple P-only or PI controllers for each diagonal loop
% Example: PI Controllers
K11 = pid(1.5, 0.1);  % for G(1,1)
K22 = pid(1.2, 0.05); % for G(2,2)
K33 = pid(2.0, 0.08); % for G(3,3)

% Controller matrix (diagonal)
K = [K11, 0, 0;
     0, K22, 0;
     0, 0, K33];

% Closed-loop system with unity feedback
T = feedback(G*K, eye(3));

% Plot closed-loop step responses
figure;
step(T, 100);
title('Closed-Loop Step Response of 3×3 MIMO System (Decentralized PID)');
