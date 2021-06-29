%%
clear
close all
clc

%% LTI systems and given parameters

LTI1.A=[1 0 2 0; 0 1 0 2; 0 0 3 0; 0 0 0 3];
LTI1.B=[2 0;0 2;3 0;0 3];
LTI1.x0=[-10;10;-1;1];

LTI2.A=[1 0 3 0; 0 1 0 3; 0 0 7 0; 0 0 0 7];
LTI2.B=[3 0; 0 3; 7 0; 0 7];
LTI2.x0=[10;10;1;1];

LTI3.A=[1 0 1 0; 0 1 0 1; 0 0 1.1 0; 0 0 0 1.1];
LTI3.B=[1 0; 0 1; 1.1 0; 0 1.1];
LTI3.x0=[10;-10;1;-1];

LTI4.A=[1 0 6 0; 0 1 0 6; 0 0 20 0; 0 0 0 20];
LTI4.B=[6 0;0 6;20 0; 0 20];
LTI4.x0=[-10;-10;-1;-1];

Tfinal=5;
umax=100;
N = 4; 

% Definition of system dimension
dim.nx=4;     %state dimension
dim.nu=2;     %input dimension
dim.N=Tfinal; %horizon

%Definition of quadratic cost function
weight.Q=eye(dim.nx);   %weight on output
weight.R=eye(dim.nu);   %weight on input

% Generation of prediction model 1
predmod1=predmodgen(LTI1,dim);            
[H1,h1]=costgen(predmod1,dim,LTI1);

% Generation of prediction model 2
predmod2=predmodgen(LTI2,dim);            
[H2,h2]=costgen(predmod2,dim,LTI2);

% Generation of prediction model 3
predmod3=predmodgen(LTI3,dim);            
[H3,h3]=costgen(predmod3,dim,LTI3);

% Generation of prediction model 4
predmod4=predmodgen(LTI4,dim);            
[H4,h4]=costgen(predmod4,dim,LTI4);

%% Constraints

%Constraints of model 1 
b_eq_1 = predmod1.T(dim.nx*(dim.N-1)+1:end,:)*LTI1.x0;
A_eq_1 = predmod1.S(dim.nx*(dim.N-1)+1:end,:);
b_eq_2 = predmod2.T(dim.nx*(dim.N-1)+1:end,:)*LTI2.x0;
A_eq_2 = predmod2.S(dim.nx*(dim.N-1)+1:end,:);
b_eq_3 = predmod3.T(dim.nx*(dim.N-1)+1:end,:)*LTI3.x0;
A_eq_3 = predmod3.S(dim.nx*(dim.N-1)+1:end,:);
b_eq_4 = predmod4.T(dim.nx*(dim.N-1)+1:end,:)*LTI4.x0;
A_eq_4 = predmod4.S(dim.nx*(dim.N-1)+1:end,:);
b_ineq = ones(2*dim.N*dim.nu,1)*umax/Tfinal;
A_ineq = blkdiag([1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1]);

%% Dual problem 
H1_d = blkdiag(H1,zeros(4,4));
H2_d = blkdiag(H2,zeros(4,4));
H3_d = blkdiag(H3,zeros(4,4));
H4_d = blkdiag(H4,zeros(4,4));
A_eq_1_d = [A_eq_1, -eye(4)];
A_eq_2_d = [A_eq_2, -eye(4)];
A_eq_3_d = [A_eq_3, -eye(4)];
A_eq_4_d = [A_eq_4, -eye(4)];
A_ineq_d =[A_ineq, zeros(20,4)];

%% Individual optimization problems
mu12 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
mu23 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
mu34 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
mu41 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
a = 1;

for i = 1:100
    opts = optimoptions('quadprog', 'Display', 'off');
%Problem 1
    h1_d = [h1, mu12'-mu41'];
    uopt1 = quadprog(2*H1_d, h1_d, A_ineq_d, b_ineq, A_eq_1_d,-b_eq_1,[],[],[],opts);
%Problem 2
    h2_d = [h2, mu23'-mu12'];
    uopt2 = quadprog(2*H2_d, h2_d, A_ineq_d, b_ineq, A_eq_2_d,-b_eq_2,[],[],[],opts);
%Problem 3
    h3_d = [h3, mu34'-mu23'];
    uopt3 = quadprog(2*H3_d, h3_d, A_ineq_d, b_ineq, A_eq_3_d,-b_eq_3,[],[],[],opts);
%Problem 4
    h4_d = [h4, mu41'-mu34'];
    uopt4 = quadprog(2*H4_d, h4_d, A_ineq_d, b_ineq, A_eq_4_d,-b_eq_4,[],[],[],opts);
    %mu update
    mu12 = mu12 + a*(uopt1(end-3:end)-uopt2(end-3:end));
    mu23 = mu23 + a*(uopt2(end-3:end)-uopt3(end-3:end));
    mu34 = mu34 + a*(uopt3(end-3:end)-uopt4(end-3:end));
    mu41 = mu41 + a*(uopt4(end-3:end)-uopt1(end-3:end));
    %Save variables for convergence analysis
    xf_1(:,i) = uopt1(end-3:end);
    xf_2(:,i) = uopt2(end-3:end);
    xf_3(:,i) = uopt3(end-3:end);
    xf_4(:,i) = uopt4(end-3:end);
    mu_save(:,i) = [mu12;mu23;mu34;mu41];
end

%% Aircarft state trajectories

u1 = uopt1(1:10);
x1 = [LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11 = x1(1:4:end);
x12 = x1(2:4:end);
x13 = x1(3:4:end);
x14 = x1(4:4:end);
u2 = uopt2(1:10);
x2 = [LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21 = x2(1:4:end);
x22 = x2(2:4:end);
x23 = x2(3:4:end);
x24 = x2(4:4:end);
u3 = uopt3(1:10);
x3 = [LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31 = x3(1:4:end);
x32 = x3(2:4:end);
x33 = x3(3:4:end);
x34 = x3(4:4:end);
u4 = uopt4(1:10);
x4 = [LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41 = x4(1:4:end);
x42 = x4(2:4:end);
x43 = x4(3:4:end);
x44 = x4(4:4:end);

for i = 1:4
    subplot(2,2,i);
    plot([0 1 2 3 4 5],x1(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x2(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x3(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x4(i:4:end))
    legend('1st Aircraft','2nd Aircraft','3rd Aircraft','4th Aircraft','Interpreter','latex')
    xlabel('$t_k$','Interpreter','latex');
    ylabel('Value','Interpreter','latex');
    title(['State ' num2str(i)],'Interpreter','latex');
end



%% Centralized solution 

Hc = blkdiag(H1,H2,H3,H4,zeros(4,4));
hc = [h1, h2, h3, h4,zeros(1,4)];
A_eq_c1 = blkdiag(A_eq_1,A_eq_2,A_eq_3,A_eq_4);
A_eq_c2 = [-eye(4);-eye(4);-eye(4);-eye(4)];
A_eq_c = [A_eq_c1, A_eq_c2];
b_eq_c = [b_eq_1;b_eq_2;b_eq_3;b_eq_4];
A_ineq_c = [blkdiag(A_ineq,A_ineq,A_ineq,A_ineq),zeros(80,4)];
b_ineq_c = ones(2*4*dim.nu*dim.N,1)*umax/Tfinal;

uc = quadprog(2*Hc,hc,A_ineq_c,b_ineq_c,A_eq_c,-b_eq_c);
xfc = uc(end-3:end);

%% Aircarft state trajectories for centralized case

u1 = uc(1:10);
x1 = [LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11 = x1(1:4:end);
x12 = x1(2:4:end);
x13 = x1(3:4:end);
x14 = x1(4:4:end);
u2 = uc(11:20);
x2 = [LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21 = x2(1:4:end);
x22 = x2(2:4:end);
x23 = x2(3:4:end);
x24 = x2(4:4:end);
u3 = uc(21:30);
x3 = [LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31 = x3(1:4:end);
x32 = x3(2:4:end);
x33 = x3(3:4:end);
x34 = x3(4:4:end);
u4 = uc(31:40);
x4 = [LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41 = x4(1:4:end);
x42 = x4(2:4:end);
x43 = x4(3:4:end);
x44 = x4(4:4:end);

for i = 1:4
    subplot(2,2,i);
    plot([0 1 2 3 4 5],x1(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x2(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x3(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x4(i:4:end))
    legend('1st Aircraft','2nd Aircraft','3rd Aircraft','4th Aircraft','Interpreter','latex')
    xlabel('$t_k$','Interpreter','latex');
    ylabel('Value','Interpreter','latex');
    title(['State ' num2str(i)],'Interpreter','latex');
end
%%

e_xf1 = 0;
e_xf2 = 0;
e_xf3 = 0;
e_xf4 = 0;

for i = 1:size(xf_1,2)-1
    e_xf1(i) = norm(xf_1(:,i)-xfc);
    e_xf2(i) = norm(xf_2(:,i)-xfc);
    e_xf3(i) = norm(xf_3(:,i)-xfc);
    e_xf4(i) = norm(xf_4(:,i)-xfc);
end

figure(1)
plot(e_xf1,'Linewidth',1.1)
hold on
plot(e_xf2,'Linewidth',1.1)
hold on
plot(e_xf3,'Linewidth',1.1)
hold on
plot(e_xf4,'Linewidth',1.1)

xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$||x_f^{1}-x_f^{Central}||$','$||x_f^{2}-x_f^{Central}||$','$||x_f^{3}-x_f^{Central}||$','$||x_f^{4}-x_f^{Central}||$','Interpreter','latex')

%% Playing with alpha


alpha = [0.5 1 2 5 10];
e_xf1 = 0;
e_xf2 = 0;
e_xf3 = 0;
e_xf4 = 0;
e_xf=[];

for j=1:5
    a=alpha(j);
    xf_1=[];
    xf_2=[];
    xf_3=[];
    xf_4=[];
    mu12 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu23 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu34 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu41 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    
    for i = 1:200
        opts = optimoptions('quadprog', 'Display', 'off');
    %Problem 1
        h1_d = [h1, mu12'-mu41'];
        uopt1 = quadprog(2*H1_d, h1_d, A_ineq_d, b_ineq, A_eq_1_d,-b_eq_1,[],[],[],opts);
    %Problem 2
        h2_d = [h2, mu23'-mu12'];
        uopt2 = quadprog(2*H2_d, h2_d, A_ineq_d, b_ineq, A_eq_2_d,-b_eq_2,[],[],[],opts);
    %Problem 3
        h3_d = [h3, mu34'-mu23'];
        uopt3 = quadprog(2*H3_d, h3_d, A_ineq_d, b_ineq, A_eq_3_d,-b_eq_3,[],[],[],opts);
    %Problem 4
        h4_d = [h4, mu41'-mu34'];
        uopt4 = quadprog(2*H4_d, h4_d, A_ineq_d, b_ineq, A_eq_4_d,-b_eq_4,[],[],[],opts);
        %mu update
        mu12 = mu12 + a*(uopt1(end-3:end)-uopt2(end-3:end))/norm(uopt1(end-3:end)-uopt2(end-3:end));
        mu23 = mu23 + a*(uopt2(end-3:end)-uopt3(end-3:end))/norm(uopt2(end-3:end)-uopt3(end-3:end));
        mu34 = mu34 + a*(uopt3(end-3:end)-uopt4(end-3:end))/norm(uopt3(end-3:end)-uopt4(end-3:end));
        mu41 = mu41 + a*(uopt4(end-3:end)-uopt1(end-3:end))/norm(uopt4(end-3:end)-uopt1(end-3:end));
        %Save variables for convergence analysis
        xf_1(:,i) = uopt1(end-3:end);
        xf_2(:,i) = uopt2(end-3:end);
        xf_3(:,i) = uopt3(end-3:end);
        xf_4(:,i) = uopt4(end-3:end);
        mu_save(:,i) = [mu12;mu23;mu34;mu41];
    end
    for p = 1:size(xf_1,2)-1
        e_xf1(j,p) = norm(xf_1(:,p)-xfc);
        e_xf2(j,p) = norm(xf_2(:,p)-xfc);
        e_xf3(j,p) = norm(xf_3(:,p)-xfc);
        e_xf4(j,p) = norm(xf_4(:,p)-xfc);
        e_xf(j,p)=(e_xf1(j,p)+e_xf2(j,p)+e_xf3(j,p)+e_xf4(j,p))/4;
    end
    
end
%%
figure(1)
plot(e_xf(1,:),'Linewidth',1.1)
hold on
plot(e_xf(2,:),'Linewidth',1.1)
hold on
plot(e_xf(3,:),'Linewidth',1.1)
hold on
plot(e_xf(4,:),'Linewidth',1.1)
hold on
plot(e_xf(5,:),'Linewidth',1.1)

xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$\alpha_k=0.5$','$\alpha_k=1$','$\alpha_k=2$','$\alpha_k=5$','$\alpha_k=10$','Interpreter','latex')

%% Time varying alpha_k


alpha = [0.5 1 2 5 10];
e_xf1 = 0;
e_xf2 = 0;
e_xf3 = 0;
e_xf4 = 0;
e_xf=[];

for j=1:5
    a=alpha(j);
    xf_1=[];
    xf_2=[];
    xf_3=[];
    xf_4=[];
    mu12 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu23 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu34 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    mu41 = zeros(4,1); %[0.1,0.1,0.1,0.1]';
    
    for i = 1:100
        opts = optimoptions('quadprog', 'Display', 'off');
    %Problem 1
        h1_d = [h1, mu12'-mu41'];
        uopt1 = quadprog(2*H1_d, h1_d, A_ineq_d, b_ineq, A_eq_1_d,-b_eq_1,[],[],[],opts);
    %Problem 2
        h2_d = [h2, mu23'-mu12'];
        uopt2 = quadprog(2*H2_d, h2_d, A_ineq_d, b_ineq, A_eq_2_d,-b_eq_2,[],[],[],opts);
    %Problem 3
        h3_d = [h3, mu34'-mu23'];
        uopt3 = quadprog(2*H3_d, h3_d, A_ineq_d, b_ineq, A_eq_3_d,-b_eq_3,[],[],[],opts);
    %Problem 4
        h4_d = [h4, mu41'-mu34'];
        uopt4 = quadprog(2*H4_d, h4_d, A_ineq_d, b_ineq, A_eq_4_d,-b_eq_4,[],[],[],opts);
        %mu update
        mu12 = mu12 + a*exp(-0.1*i)*(uopt1(end-3:end)-uopt2(end-3:end))/norm(uopt1(end-3:end)-uopt2(end-3:end));
        mu23 = mu23 + a*exp(-0.1*i)*(uopt2(end-3:end)-uopt3(end-3:end))/norm(uopt2(end-3:end)-uopt3(end-3:end));
        mu34 = mu34 + a*exp(-0.1*i)*(uopt3(end-3:end)-uopt4(end-3:end))/norm(uopt3(end-3:end)-uopt4(end-3:end));
        mu41 = mu41 + a*exp(-0.1*i)*(uopt4(end-3:end)-uopt1(end-3:end))/norm(uopt4(end-3:end)-uopt1(end-3:end));
        %Save variables for convergence analysis
        xf_1(:,i) = uopt1(end-3:end);
        xf_2(:,i) = uopt2(end-3:end);
        xf_3(:,i) = uopt3(end-3:end);
        xf_4(:,i) = uopt4(end-3:end);
        mu_save(:,i) = [mu12;mu23;mu34;mu41];
    end
    for p = 1:size(xf_1,2)-1
        e_xf1(j,p) = norm(xf_1(:,p)-xfc);
        e_xf2(j,p) = norm(xf_2(:,p)-xfc);
        e_xf3(j,p) = norm(xf_3(:,p)-xfc);
        e_xf4(j,p) = norm(xf_4(:,p)-xfc);
        e_xf(j,p)=(e_xf1(j,p)+e_xf2(j,p)+e_xf3(j,p)+e_xf4(j,p))/4;
    end
    
end
%%
figure(1)
plot(e_xf(1,:),'Linewidth',1.1)
hold on
plot(e_xf(2,:),'Linewidth',1.1)
hold on
plot(e_xf(3,:),'Linewidth',1.1)
hold on
plot(e_xf(4,:),'Linewidth',1.1)
hold on
plot(e_xf(5,:),'Linewidth',1.1)

xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$\alpha_k=0.5$','$\alpha_k=1$','$\alpha_k=2$','$\alpha_k=5$','$\alpha_k=10$','Interpreter','latex')

%% Nesterov 


Aeq_ = [A_eq_1 -A_eq_2 zeros(dim.nx,dim.nu*dim.N) zeros(dim.nx,dim.nu*dim.N);
        zeros(dim.nx,dim.nu*dim.N) A_eq_2 -A_eq_3 zeros(dim.nx,dim.nu*dim.N);
        zeros(dim.nx,dim.nu*dim.N) zeros(dim.nx,dim.nu*dim.N) A_eq_3 -A_eq_4;
        -A_eq_1 zeros(dim.nx,dim.nu*dim.N) zeros(dim.nx,dim.nu*dim.N) A_eq_4];
beq_ = [b_eq_2 - b_eq_1; b_eq_3 - b_eq_2; b_eq_4 - b_eq_3; b_eq_1 - b_eq_4];

Ain_ = blkdiag(A_ineq, A_ineq, A_ineq, A_ineq);
bin_ = repmat(b_ineq, 4,1);

Anest = [Aeq_; Ain_]; bnest = [beq_; bin_];
Hnest = blkdiag(H1,H2,H3,H4); hnest = [h1, h2, h3, h4]';

L = norm(Anest/Hnest*Anest'/2,2);
%initialize
% Constraints on xfi = xfj
l12 = zeros(4,1); l23 = zeros(4,1); l34 = zeros(4,1); l41 = zeros(4,1);
ls = [[l12;l23;l34;l41] , [l12;l23;l34;l41]];
% Constraints on ui <= umax/Tfinal and -ui <= umax/Tfinal
mu1 = zeros(20,1); mu2 = zeros(20,1); mu3 = zeros(20,1); mu4 = zeros(20,1);
mus = [[mu1;mu2;mu3;mu4] , [mu1;mu2;mu3;mu4]];
tic
for i = 1:40
    opts = optimoptions('quadprog', 'Display', 'off');
%Problem 1
    h1d = h1 + (l12'-l41')*A_eq_1 + mu1'*A_ineq;
    u1 = quadprog(2*H1,h1d,[],[],[],[],[],[],[],opts);
    xf1 = A_eq_1*u1+b_eq_1; xf1 = xf1(:); xf1v(:,i) = xf1;
%Problem 2
    h2d = h2 + (l23'-l12')*A_eq_2 + mu2'*A_ineq;
    u2 = quadprog(2*H2,h2d,[],[],[],[],[],[],[],opts);
    xf2 = A_eq_2*u2+b_eq_2; xf2 = xf2(:); xf2v(:,i) = xf2;
%Problem 3
    h3d = h3 + (l34'-l23')*A_eq_3 + mu3'*A_ineq;
    u3 = quadprog(2*H3,h3d,[],[],[],[],[],[],[],opts);
    xf3 = A_eq_3*u3+b_eq_3; xf3 = xf3(:); xf3v(:,i) = xf3;
%Problem 4
    h4d = h4 + (l41'-l34')*A_eq_4 + mu4'*A_ineq;
    u4 = quadprog(2*H4,h4d,[],[],[],[],[],[],[],opts);
    xf4 = A_eq_4*u4+b_eq_4; xf4 = xf4(:); xf4v(:,i) = xf4;
%Dual variables update
    
    zm1 = [ls(:,i); mus(:,i)];
    z = [ls(:,i+1); mus(:,i+1)];
    
    % alph = (1-sqrt(i))/(1+sqrt(i));
    alph = .75*(i-1)/(i+2);
    vk = z + alph*(z-zm1);
    
    gradf = 1/2 * Anest/Hnest*(Anest'*vk + hnest) + bnest;
    
    zp1 = vk - 1/L*gradf;
    
    ls(:,i+2) = zp1(1:16); % ls = [l-1 , l0, l1, l2, ...]
    mus(:,i+2) = max(zp1(17:end), 0); % mus = [mu-1, mu0, mu1, mu2, ...]

    l12 = ls(1:4,i+2); l23 = ls(5:8,i+2); l34 = ls(9:12,i+2); l41 = ls(13:16,i+2);   
    mu1 = mus(1:20,i+2); mu2 = mus(21:40,i+2); mu3 = mus(41:60,i+2); mu4 = mus(61:80,i+2); 
end
%%

e_xf1 = 0;
for i = 1:size(xf1v,2)-1
    e_xf1(i) = norm(xf1v(:,i)-xfc);
end

e_xf2 = 0;
for i = 1:size(xf1v,2)-1
    e_xf2(i) = norm(xf2v(:,i)-xfc);
end

e_xf3 = 0;
for i = 1:size(xf1v,2)-1
    e_xf3(i) = norm(xf3v(:,i)-xfc);
end
e_xf4 = 0;
for i = 1:size(xf1v,2)-1
    e_xf4(i) = norm(xf4v(:,i)-xfc);
end

figure(2)
plot(e_xf1,'Linewidth',1.1)
hold on
plot(e_xf2,'Linewidth',1.1)
hold on
plot(e_xf3,'Linewidth',1.1)
hold on
plot(e_xf4,'Linewidth',1.1)

xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$||x_f^{1}-x_f^{Central}||$','$||x_f^{2}-x_f^{Central}||$','$||x_f^{3}-x_f^{Central}||$','$||x_f^{4}-x_f^{Central}||$','Interpreter','latex')

%% Consensus 
W=[0.75 0.25 0 0;0.25 0.5 0.25 0; 0 0.25 0.5 0.25;0 0 0.25 0.75];
Phi = [1 5 10 20 50 100]; 
iter=1000;

for p= 1:6
    phi=Phi(p);
    Wphi=W^phi;
    clear xf 
    clear g 
    xf{1}=zeros(4,iter+1);
    xf{2}=zeros(4,iter+1);
    xf{3}=zeros(4,iter+1);
    xf{4}=zeros(4,iter+1);


    for k = 1:iter
        a=0.2;
        opts = optimoptions('quadprog', 'Display', 'off');
        %Problem 1    
        [uopt1,~,~,~,lambda1] = quadprog(2*H1, h1, A_ineq, b_ineq, A_eq_1,xf{1}(:,k)-b_eq_1,[],[],[],opts);
        g{1}(:,k)=lambda1.eqlin;
        %Problem 2 
        [uopt2,~,~,~,lambda2] = quadprog(2*H2, h2, A_ineq, b_ineq, A_eq_2,xf{2}(:,k)-b_eq_2,[],[],[],opts);
        g{2}(:,k)=lambda2.eqlin;
        %Problem 3    
        [uopt3,~,~,~,lambda3] = quadprog(2*H3, h3, A_ineq, b_ineq, A_eq_3,xf{3}(:,k)-b_eq_3,[],[],[],opts);
        g{3}(:,k)=lambda3.eqlin;
        %Problem 4   
        [uopt4,~,~,~,lambda4] = quadprog(2*H4, h4, A_ineq, b_ineq, A_eq_4,xf{4}(:,k)-b_eq_4,[],[],[],opts);
        g{4}(:,k) =lambda4.eqlin;

        %update

        for i=1:4
            for j=1:4
                xf{i}(:,k+1)=xf{i}(:,k+1)+Wphi(i,j)*(xf{j}(:,k)+a/k*g{j}(:,k));
            end
        end

    end
    
    for i=1:iter
        error_x1(i)=norm(xf{1}(:,i)-xfc);
        error_x2(i)=norm(xf{2}(:,i)-xfc);
        error_x3(i)=norm(xf{3}(:,i)-xfc);
        error_x4(i)=norm(xf{4}(:,i)-xfc);
        error_xf(p,i)= error_x1(i)+error_x2(i)+error_x3(i)+error_x4(i);
    end
end 
%%
figure(2)

plot(error_xf(1,:),'Linewidth',1.1)
hold on
plot(error_xf(2,:),'Linewidth',1.1)
hold on
plot(error_xf(3,:),'Linewidth',1.1)
hold on
plot(error_xf(4,:),'Linewidth',1.1)
hold on
plot(error_xf(5,:),'Linewidth',1.1)


xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$\varphi=1$','$\varphi=5$','$\varphi=10$','$\varphi=20$','$\varphi=50$','Interpreter','latex')

%% ADMM

%initialize
xf0 = [0;0;0;0]; %optimal solution from combined -5.9566   -3.7671   -7.3848   -4.6138
y0 = [0;0;0;0];
rho = 1;
iter = 1000;

xf = zeros(N,iter);
xf(:,1) = xf0;
y1 = zeros(dim.nx, iter);
y1(:,1) = y0;
y2 = zeros(dim.nx, iter);
y2(:,1) = y0;
y3 = zeros(dim.nx, iter);
y3(:,1) = y0;
y4 = zeros(dim.nx, iter);
y4(:,1) = y0;
uopt1_ = zeros(iter, dim.nu*dim.N);
uopt2_ = zeros(iter, dim.nu*dim.N);
uopt3_ = zeros(iter, dim.nu*dim.N);
uopt4_ = zeros(iter, dim.nu*dim.N);
ubar = zeros(N,iter);
ubar(:,1) = y0;

%% ADMM algorithm with quadprog
for k = 1:iter    
    %1) Update x's
    %u1(k+1)
    uopt1 = quadprog(2*(H1 + rho/2*A_eq_1'*A_eq_1),(h1+y1(:,k)'*A_eq_1+rho*(b_eq_1 - xf(:,k))'*A_eq_1),A_ineq, b_ineq,[],[],[],[],[],opts);
    uopt1_(k,:) = uopt1;
    %u2(k+1)
    uopt2 = quadprog(2*(H2 + rho/2*A_eq_2'*A_eq_2),(h2+y2(:,k)'*A_eq_2+rho*(b_eq_2 - xf(:,k))'*A_eq_2),A_ineq, b_ineq,[],[],[],[],[],opts);
    uopt2_(k,:) = uopt2;
    %u3(k+1)
    uopt3 = quadprog(2*(H3 + rho/2*A_eq_3'*A_eq_3),(h3+y3(:,k)'*A_eq_3+rho*(b_eq_3 - xf(:,k))'*A_eq_3),A_ineq, b_ineq,[],[],[],[],[],opts);
    uopt3_(k,:) = uopt3;
    %u4(k+1)
    uopt4 = quadprog(2*(H4 + rho/2*A_eq_4'*A_eq_4),(h4+y4(:,k)'*A_eq_4+rho*(b_eq_4 - xf(:,k))'*A_eq_4),A_ineq, b_ineq,[],[],[],[],[],opts);
    uopt4_(k,:) = uopt4;
    %2) Update xf
    xf(:,k+1) = 1/N*((A_eq_1*uopt1 + b_eq_1 + 1/rho*y1(:,k))+(A_eq_2*uopt2 + b_eq_2 + 1/rho*y2(:,k))+(A_eq_3*uopt3 + b_eq_3 + 1/rho*y3(:,k))+(A_eq_4*uopt4 + b_eq_4 + 1/rho*y4(:,k)));
    %3) Update y's
    y1(:,k+1) = y1(:,k) + rho*(A_eq_1*uopt1 + b_eq_1 - xf(:,k+1));
    y2(:,k+1) = y2(:,k) + rho*(A_eq_2*uopt2 + b_eq_2 - xf(:,k+1));
    y3(:,k+1) = y3(:,k) + rho*(A_eq_3*uopt3 + b_eq_3 - xf(:,k+1));
    y4(:,k+1) = y4(:,k) + rho*(A_eq_4*uopt4 + b_eq_4 - xf(:,k+1));
end
%% Error sequence plot 
e_xf = 0;
for i = 1:size(xf,2)-1;
    e_xf(i) = norm(xf(:,i)-xfc);
end
figure(3)
loglog(e_xf,'Linewidth',1.1)
grid on
ylim([0.0001  10])
xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$||x_f^{k}-x_f^{Central}||$ with $\rho=1$','Interpreter','latex')
%% Trajectories ADMM


u1 = uopt1;
x1 = [LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11 = x1(1:4:end);
x12 = x1(2:4:end);
x13 = x1(3:4:end);
x14 = x1(4:4:end);
u2 = uopt2;
x2 = [LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21 = x2(1:4:end);
x22 = x2(2:4:end);
x23 = x2(3:4:end);
x24 = x2(4:4:end);
u3 = uopt3;
x3 = [LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31 = x3(1:4:end);
x32 = x3(2:4:end);
x33 = x3(3:4:end);
x34 = x3(4:4:end);
u4 = uopt4;
x4 = [LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41 = x4(1:4:end);
x42 = x4(2:4:end);
x43 = x4(3:4:end);
x44 = x4(4:4:end);

for i = 1:4
    subplot(2,2,i);
    plot([0 1 2 3 4 5],x1(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x2(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x3(i:4:end))
    hold on
    plot([0 1 2 3 4 5],x4(i:4:end))
    legend('1st Aircraft','2nd Aircraft','3rd Aircraft','4th Aircraft','Interpreter','latex')
    xlabel('$t_k$','Interpreter','latex');
    ylabel('Value','Interpreter','latex');
    title(['State ' num2str(i)],'Interpreter','latex');
end

%% plot results 

figure(4)
plot(xf(1,:),'Linewidth',1.1)
hold on 
plot(xf(2,:),'Linewidth',1.1)
hold on
plot(xf(3,:),'Linewidth',1.1)
hold on
plot(xf(4,:),'Linewidth',1.1)

legend('$x_{f,1}$','$x_{f,2}$','$x_{f,3}$','$x_{f,4}$','Interpreter','latex')
xlim([0 100])
xlabel('Number of iterations','Interpreter','latex')
ylabel('$x_{f,i}$ State Value','Interpreter','latex')

%%
xf0 = [0;0;0;0]; 
y0 = [0;0;0;0];
r = [5 2 1 0.5 0.2];
iter = 100;
N=4;
e_xf=[];

xf = zeros(N,iter);
xf(:,1) = xf0;
y1 = zeros(dim.nx, iter);
y1(:,1) = y0;
y2 = zeros(dim.nx, iter);
y2(:,1) = y0;
y3 = zeros(dim.nx, iter);
y3(:,1) = y0;
y4 = zeros(dim.nx, iter);
y4(:,1) = y0;
uopt1_ = zeros(iter, dim.nu*dim.N);
uopt2_ = zeros(iter, dim.nu*dim.N);
uopt3_ = zeros(iter, dim.nu*dim.N);
uopt4_ = zeros(iter, dim.nu*dim.N);
ubar = zeros(N,iter);
ubar(:,1) = y0;

for j=1:5
    rho=r(j);
    for k = 1:iter

        %1) Update x's
        %u1(k+1)
        uopt1 = quadprog(2*(H1 + rho/2*A_eq_1'*A_eq_1),(h1+y1(:,k)'*A_eq_1+rho*(b_eq_1 - xf(:,k))'*A_eq_1),A_ineq, b_ineq,[],[],[],[],[],opts);
        uopt1_(k,:) = uopt1;
        %u2(k+1)
        uopt2 = quadprog(2*(H2 + rho/2*A_eq_2'*A_eq_2),(h2+y2(:,k)'*A_eq_2+rho*(b_eq_2 - xf(:,k))'*A_eq_2),A_ineq, b_ineq,[],[],[],[],[],opts);
        uopt2_(k,:) = uopt2;
        %u3(k+1)
        uopt3 = quadprog(2*(H3 + rho/2*A_eq_3'*A_eq_3),(h3+y3(:,k)'*A_eq_3+rho*(b_eq_3 - xf(:,k))'*A_eq_3),A_ineq, b_ineq,[],[],[],[],[],opts);
        uopt3_(k,:) = uopt3;
        %u4(k+1)
        uopt4 = quadprog(2*(H4 + rho/2*A_eq_4'*A_eq_4),(h4+y4(:,k)'*A_eq_4+rho*(b_eq_4 - xf(:,k))'*A_eq_4),A_ineq, b_ineq,[],[],[],[],[],opts);
        uopt4_(k,:) = uopt4;
        %2) Update xf
        xf(:,k+1) = 1/N*((A_eq_1*uopt1 + b_eq_1 + 1/rho*y1(:,k))+(A_eq_2*uopt2 + b_eq_2 + 1/rho*y2(:,k))+(A_eq_3*uopt3 + b_eq_3 + 1/rho*y3(:,k))+(A_eq_4*uopt4 + b_eq_4 + 1/rho*y4(:,k)));
        %3) Update y's
        y1(:,k+1) = y1(:,k) + rho*(A_eq_1*uopt1 + b_eq_1 - xf(:,k+1));
        y2(:,k+1) = y2(:,k) + rho*(A_eq_2*uopt2 + b_eq_2 - xf(:,k+1));
        y3(:,k+1) = y3(:,k) + rho*(A_eq_3*uopt3 + b_eq_3 - xf(:,k+1));
        y4(:,k+1) = y4(:,k) + rho*(A_eq_4*uopt4 + b_eq_4 - xf(:,k+1));
    end
    
    
    for i = 1:size(xf,2)-1
        e_xf(i,j) = norm(xf(:,i)-xfc);
    end
end
%%
figure(5)
plot(e_xf(:,1),'Linewidth',1.1)

hold on
plot(e_xf(:,2),'Linewidth',1.1)
hold on
plot(e_xf(:,3),'Linewidth',1.1)
hold on
plot(e_xf(:,4),'Linewidth',1.1)
hold on
plot(e_xf(:,5),'Linewidth',1.1)
hold on
ylim([0 3])
xlim([0 100])
xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$||x_f^{k}-x_f^{Central}||$ with $\rho=5$','$||x_f^{k}-x_f^{Central}||$ with $\rho=2$','$||x_f^{k}-x_f^{Central}||$ with $\rho=1$','$||x_f^{k}-x_f^{Central}||$ with $\rho=0.5$','$||x_f^{k}-x_f^{Central}||$ with $\rho=0.2$','Interpreter','latex')



