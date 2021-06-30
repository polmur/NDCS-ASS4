%% Assignment 4 NDCS

clear
close all
clc

%% Definition of each of the LTI systems+generation of Hi and gi matrices 

Tf=5;
umax=100;
N=4; 

dim.nx=4;     %number of states
dim.nu=2;     %number of inputs
dim.N=Tf;     %horizon

weight.Q=eye(dim.nx);  
weight.R=eye(dim.nu); 

LTI1.A=[1 0 2 0; 0 1 0 2; 0 0 3 0; 0 0 0 3];
LTI1.B=[2 0;0 2;3 0;0 3];
LTI1.x0=[-10;10;-1;1];
predmod1=predmodgen(LTI1,dim);            
[H1,g1]=costgen(predmod1,dim,LTI1);

LTI2.A=[1 0 3 0; 0 1 0 3; 0 0 7 0; 0 0 0 7];
LTI2.B=[3 0; 0 3; 7 0; 0 7];
LTI2.x0=[10;10;1;1];
predmod2=predmodgen(LTI2,dim);            
[H2,g2]=costgen(predmod2,dim,LTI2);

LTI3.A=[1 0 1 0; 0 1 0 1; 0 0 1.1 0; 0 0 0 1.1];
LTI3.B=[1 0; 0 1; 1.1 0; 0 1.1];
LTI3.x0=[10;-10;1;-1];
predmod3=predmodgen(LTI3,dim);            
[H3,g3]=costgen(predmod3,dim,LTI3);

LTI4.A=[1 0 6 0; 0 1 0 6; 0 0 20 0; 0 0 0 20];
LTI4.B=[6 0;0 6;20 0; 0 20];
LTI4.x0=[-10;-10;-1;-1];
predmod4=predmodgen(LTI4,dim);            
[H4,g4]=costgen(predmod4,dim,LTI4);


%% Generation of constraints  
%Equality Constraints 1
beq1=predmod1.T(dim.nx*(dim.N-1)+1:end,:)*LTI1.x0;
Aeq1=predmod1.S(dim.nx*(dim.N-1)+1:end,:);
%Equality Constraints 2
beq2=predmod2.T(dim.nx*(dim.N-1)+1:end,:)*LTI2.x0;
Aeq2=predmod2.S(dim.nx*(dim.N-1)+1:end,:);
%Equality Constraints 3
beq3=predmod3.T(dim.nx*(dim.N-1)+1:end,:)*LTI3.x0;
Aeq3=predmod3.S(dim.nx*(dim.N-1)+1:end,:);
%Equality Constraints 4
beq4=predmod4.T(dim.nx*(dim.N-1)+1:end,:)*LTI4.x0;
Aeq4=predmod4.S(dim.nx*(dim.N-1)+1:end,:);
%Inequality constraints
bin=ones(2*dim.N*dim.nu,1)*umax/Tf;
Ain=blkdiag([1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1],[1;-1]);

%% Extension of the constraint matrices to include xf as opt variable 
H1e=blkdiag(H1,zeros(4,4));
H2e=blkdiag(H2,zeros(4,4));
H3e=blkdiag(H3,zeros(4,4));
H4e=blkdiag(H4,zeros(4,4));

Aeq1e=[Aeq1, -eye(4)];
Aeq2e=[Aeq2, -eye(4)];
Aeq3e=[Aeq3, -eye(4)];
Aeq4e=[Aeq4, -eye(4)];

Aine =[Ain, zeros(20,4)];

%% Dual Decomposition approach
%Initialize values of muij
mu_12=zeros(4,1); 
mu_23=zeros(4,1); 
mu_34=zeros(4,1); 
mu_41=zeros(4,1);
alphak=1;

% We solve each of the four different problems in parallel and then update
% the corresponding muij globally

for i=1:100
    opts=optimoptions('quadprog', 'Display', 'off');
    g1_d=[g1, mu_12'-mu_41'];
    uopt1=quadprog(2*H1e, g1_d, Aine, bin, Aeq1e,-beq1,[],[],[],opts);
    g2_d=[g2, mu_23'-mu_12'];
    uopt2=quadprog(2*H2e, g2_d, Aine, bin, Aeq2e,-beq2,[],[],[],opts);
    g3_d=[g3, mu_34'-mu_23'];
    uopt3=quadprog(2*H3e, g3_d, Aine, bin, Aeq3e,-beq3,[],[],[],opts);
    g4_d=[g4, mu_41'-mu_34'];
    uopt4=quadprog(2*H4e, g4_d, Aine, bin, Aeq4e,-beq4,[],[],[],opts);
    %Update of the dual variables, xf is now updated in the fourth last
    %positions of each of the uopti
    mu_12=mu_12+alphak*(uopt1(11:14)-uopt2(11:14));
    mu_23=mu_23+alphak*(uopt2(11:14)-uopt3(11:14));
    mu_34=mu_34+alphak*(uopt3(11:14)-uopt4(11:14));
    mu_41=mu_41+alphak*(uopt4(11:14)-uopt1(11:14));
    
    xf_1(:,i)=uopt1(11:14);
    xf_2(:,i)=uopt2(11:14);
    xf_3(:,i)=uopt3(11:14);
    xf_4(:,i)=uopt4(11:14);
end

%% Simulate the evolution of the states of each of the aircrafts

u1=uopt1(1:10);
x1=[LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11=x1(1:4:end);
x12=x1(2:4:end);
x13=x1(3:4:end);
x14=x1(4:4:end);
u2=uopt2(1:10);
x2=[LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21=x2(1:4:end);
x22=x2(2:4:end);
x23=x2(3:4:end);
x24=x2(4:4:end);
u3=uopt3(1:10);
x3=[LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31=x3(1:4:end);
x32=x3(2:4:end);
x33=x3(3:4:end);
x34=x3(4:4:end);
u4=uopt4(1:10);
x4=[LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41=x4(1:4:end);
x42=x4(2:4:end);
x43=x4(3:4:end);
x44=x4(4:4:end);

for i=1:4
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



%% Optimal Centralized Solution 

Hc=blkdiag(H1,H2,H3,H4,zeros(4,4));
gc=[g1, g2, g3, g4,zeros(1,4)];

A_eq_c1=blkdiag(Aeq1,Aeq2,Aeq3,Aeq4);
A_eq_c2=[-eye(4);-eye(4);-eye(4);-eye(4)];
A_eq_c=[A_eq_c1, A_eq_c2];
b_eq_c=[beq1;beq2;beq3;beq4];

A_ineq_c=[blkdiag(Ain,Ain,Ain,Ain),zeros(80,4)];
b_ineq_c=ones(80,1)*umax/Tf;

uc=quadprog(2*Hc,gc,A_ineq_c,b_ineq_c,A_eq_c,-b_eq_c);
xfc=uc(41:44);

%% Aircarft state trajectories for centralized case

u1=uc(1:10);
x1=[LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11=x1(1:4:end);
x12=x1(2:4:end);
x13=x1(3:4:end);
x14=x1(4:4:end);
u2=uc(11:20);
x2=[LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21=x2(1:4:end);
x22=x2(2:4:end);
x23=x2(3:4:end);
x24=x2(4:4:end);
u3=uc(21:30);
x3=[LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31=x3(1:4:end);
x32=x3(2:4:end);
x33=x3(3:4:end);
x34=x3(4:4:end);
u4=uc(31:40);
x4=[LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41=x4(1:4:end);
x42=x4(2:4:end);
x43=x4(3:4:end);
x44=x4(4:4:end);

for i=1:4
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

e_xf1=0;
e_xf2=0;
e_xf3=0;
e_xf4=0;

for i=1:size(xf_1,2)-1
    e_xf1(i)=norm(xf_1(:,i)-xfc);
    e_xf2(i)=norm(xf_2(:,i)-xfc);
    e_xf3(i)=norm(xf_3(:,i)-xfc);
    e_xf4(i)=norm(xf_4(:,i)-xfc);
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

alpha=[0.5 1 2 5 10];
e_xf1=0;
e_xf2=0;
e_xf3=0;
e_xf4=0;
e_xf=[];

for j=1:5
    alphak=alpha(j);
    xf_1=[];
    xf_2=[];
    xf_3=[];
    xf_4=[];
    mu_12=zeros(4,1);
    mu_23=zeros(4,1); 
    mu_34=zeros(4,1); 
    mu_41=zeros(4,1); 
    
    for i=1:200
        opts=optimoptions('quadprog', 'Display', 'off');
        g1_d=[g1, mu_12'-mu_41'];
        uopt1=quadprog(2*H1e, g1_d, Aine, bin, Aeq1e,-beq1,[],[],[],opts);
        g2_d=[g2, mu_23'-mu_12'];
        uopt2=quadprog(2*H2e, g2_d, Aine, bin, Aeq2e,-beq2,[],[],[],opts);
        g3_d=[g3, mu_34'-mu_23'];
        uopt3=quadprog(2*H3e, g3_d, Aine, bin, Aeq3e,-beq3,[],[],[],opts);
        g4_d=[g4, mu_41'-mu_34'];
        uopt4=quadprog(2*H4e, g4_d, Aine, bin, Aeq4e,-beq4,[],[],[],opts);
        %Update of the dual variables, xf is now updated in the fourth last
        %positions of each of the uopti
        mu_12=mu_12+alphak*(uopt1(11:14)-uopt2(11:14));
        mu_23=mu_23+alphak*(uopt2(11:14)-uopt3(11:14));
        mu_34=mu_34+alphak*(uopt3(11:14)-uopt4(11:14));
        mu_41=mu_41+alphak*(uopt4(11:14)-uopt1(11:14));

        xf_1(:,i)=uopt1(11:14);
        xf_2(:,i)=uopt2(11:14);
        xf_3(:,i)=uopt3(11:14);
        xf_4(:,i)=uopt4(11:14);
    end
    
    for p=1:4
        e_xf1(j,p)=norm(xf_1(:,p)-xfc);
        e_xf2(j,p)=norm(xf_2(:,p)-xfc);
        e_xf3(j,p)=norm(xf_3(:,p)-xfc);
        e_xf4(j,p)=norm(xf_4(:,p)-xfc);
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

alpha=[0.5 1 2 5 10];
e_xf1=0;
e_xf2=0;
e_xf3=0;
e_xf4=0;
e_xf=[];

for j=1:5
    alphak=alpha(j);
    xf_1=[];
    xf_2=[];
    xf_3=[];
    xf_4=[];
    mu_12=zeros(4,1);
    mu_23=zeros(4,1);
    mu_34=zeros(4,1); 
    mu_41=zeros(4,1);
    
    for i=1:100
        opts=optimoptions('quadprog', 'Display', 'off');
        g1_d=[g1, mu_12'-mu_41'];
        uopt1=quadprog(2*H1e, g1_d, Aine, bin, Aeq1e,-beq1,[],[],[],opts);
        g2_d=[g2, mu_23'-mu_12'];
        uopt2=quadprog(2*H2e, g2_d, Aine, bin, Aeq2e,-beq2,[],[],[],opts);
        g3_d=[g3, mu_34'-mu_23'];
        uopt3=quadprog(2*H3e, g3_d, Aine, bin, Aeq3e,-beq3,[],[],[],opts);
        g4_d=[g4, mu_41'-mu_34'];
        uopt4=quadprog(2*H4e, g4_d, Aine, bin, Aeq4e,-beq4,[],[],[],opts);
        %Update of the dual variables, xf is now updated in the fourth last
        %positions of each of the uopti
        mu_12=mu_12+alphak*exp(-0.1*i)*(uopt1(11:14)-uopt2(11:14))/norm(uopt1(11:14)-uopt2(11:14));
        mu_23=mu_23+alphak*exp(-0.1*i)*(uopt2(11:14)-uopt3(11:14))/norm(uopt2(11:14)-uopt3(11:14));
        mu_34=mu_34+alphak*exp(-0.1*i)*(uopt3(11:14)-uopt4(11:14))/norm(uopt3(11:14)-uopt4(11:14));
        mu_41=mu_41+alphak*exp(-0.1*i)*(uopt4(11:14)-uopt1(11:14))/norm(uopt4(11:14)-uopt1(11:14));
        
        xf_1(:,i)=uopt1(11:14);
        xf_2(:,i)=uopt2(11:14);
        xf_3(:,i)= uopt3(11:14);
        xf_4(:,i)=uopt4(11:14);
        mu_save(:,i)=[mu_12;mu_23;mu_34;mu_41];
    end
    for p=1:4
        e_xf1(j,p)=norm(xf_1(:,p)-xfc);
        e_xf2(j,p)=norm(xf_2(:,p)-xfc);
        e_xf3(j,p)=norm(xf_3(:,p)-xfc);
        e_xf4(j,p)=norm(xf_4(:,p)-xfc);
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


Aeqm=[Aeq1 -Aeq2 zeros(4,10) zeros(4,10);zeros(4,10) Aeq2 -Aeq3 zeros(4,10);zeros(4,10) zeros(4,10) Aeq3 -Aeq4;-Aeq1 zeros(4,10) zeros(4,10) Aeq4];
beqn=[beq2-beq1; beq3-beq2; beq4-beq3; beq1-beq4];

Ainn=blkdiag(Ain, Ain, Ain, Ain);
binn=[bineq;bineq;bineq;bineq];

An=[Aeqm; Ainn];
bn=[beqn; binn];
Hn=blkdiag(H1,H2,H3,H4); gn=[g1, g2, g3, g4]';

L=norm(An/Hn*An'/2,2);

l12=zeros(4,1); 
l23=zeros(4,1);
l34=zeros(4,1);
l41=zeros(4,1);

l=[[l12;l23;l34;l41] , [l12;l23;l34;l41]];

mu1=zeros(20,1);
mu2=zeros(20,1);
mu3=zeros(20,1);
mu4=zeros(20,1);
mu=[[mu1;mu2;mu3;mu4] , [mu1;mu2;mu3;mu4]];

for i=1:40
    opts=optimoptions('quadprog', 'Display', 'off');

    h1d=g1+(l12'-l41')*Aeq1+mu1'*Ain;
    u1=quadprog(2*H1,h1d,[],[],[],[],[],[],[],opts);
    xf1=Aeq1*u1+beq1;
    xf1n(:,i)=xf1;

    h2d=g2+(l23'-l12')*Aeq2+mu2'*Ain;
    u2=quadprog(2*H2,h2d,[],[],[],[],[],[],[],opts);
    xf2=Aeq2*u2+beq2;
    xf2n(:,i)=xf2;

    h3d=g3+(l34'-l23')*Aeq3+mu3'*Ain;
    u3=quadprog(2*H3,h3d,[],[],[],[],[],[],[],opts);
    xf3=Aeq3*u3+beq3;    
    xf3n(:,i)=xf3;

    h4d=g4+(l41'-l34')*Aeq4+mu4'*Ain;
    u4=quadprog(2*H4,h4d,[],[],[],[],[],[],[],opts);
    xf4=Aeq4*u4+beq4;     
    xf4n(:,i)=xf4;
    
    zm1=[l(:,i);mu(:,i)];
    z=[l(:,i+1);mu(:,i+1)];    
    
    alpha=0.75*(i-1)/(i+2);
    vk=z+alpha*(z-zm1);
    
    grad=1/2*An/Hn*(An'*vk+gn)+bn;
    
    zp1=vk-1/L*grad;
    
    l(:,i+2)=zp1(1:16); 
    mu(:,i+2)=max(zp1(17:end), 0); 

    l12=l(1:4,i+2);
    l23=l(5:8,i+2);
    l34=l(9:12,i+2);
    l41=l(13:16,i+2);
    
    mu1=mu(1:20,i+2);
    mu2=mu(21:40,i+2);
    mu3=mu(41:60,i+2);
    mu4=mu(61:80,i+2); 
end
%%

e_xf1=0;
e_xf2=0;
e_xf3=0;
e_xf4=0;

for i=1:size(xf1n,2)-1
    e_xf1(i)=norm(xf1n(:,i)-xfc);
    e_xf2(i)=norm(xf2n(:,i)-xfc);
    e_xf3(i)=norm(xf3n(:,i)-xfc);
    e_xf4(i)=norm(xf4n(:,i)-xfc);
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
Phi=[1 5 10 20 50 100]; 
iterations=1000;

for p= 1:6
    phi=Phi(p);
    Wphi=W^phi;
    clear xf 
    clear g 
    xf{1}=zeros(4,iterations+1);
    xf{2}=zeros(4,iterations+1);
    xf{3}=zeros(4,iterations+1);
    xf{4}=zeros(4,iterations+1);


    for k=1:iterations
        alphak=0.2;
        opts=optimoptions('quadprog', 'Display', 'off');            
        [uopt1,~,~,~,lambda1]=quadprog(2*H1, g1, Ain, bin, Aeq1,xf{1}(:,k)-beq1,[],[],[],opts);
        % we can obtain the subgradient with the lambdas
        g{1}(:,k)=lambda1.eqlin;         
        [uopt2,~,~,~,lambda2]=quadprog(2*H2, g2, Ain, bin, Aeq2,xf{2}(:,k)-beq2,[],[],[],opts);
        g{2}(:,k)=lambda2.eqlin;          
        [uopt3,~,~,~,lambda3]=quadprog(2*H3, g3, Ain, bin, Aeq3,xf{3}(:,k)-beq3,[],[],[],opts);
        g{3}(:,k)=lambda3.eqlin;           
        [uopt4,~,~,~,lambda4]=quadprog(2*H4, g4, Ain, bin, Aeq4,xf{4}(:,k)-beq4,[],[],[],opts);
        g{4}(:,k) =lambda4.eqlin;

        %Consensus Update
        for i=1:4
            for j=1:4
                xf{i}(:,k+1)=xf{i}(:,k+1)+Wphi(i,j)*(xf{j}(:,k)+alphak/k*g{j}(:,k));
            end
        end

    end
    
    for i=1:iterations
        error_x1(i)=norm(xf{1}(:,i)-xfc);
        error_x2(i)=norm(xf{2}(:,i)-xfc);
        error_x3(i)=norm(xf{3}(:,i)-xfc);
        error_x4(i)=norm(xf{4}(:,i)-xfc);
        error_xf(p,i)= (error_x1(i)+error_x2(i)+error_x3(i)+error_x4(i))/4;
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

rho=1;
iterations=1000;
xf=zeros(N,iterations);
y1=zeros(4, iterations);
y2=zeros(4, iterations);
y3=zeros(4, iterations);
y4=zeros(4, iterations);

uopt1_admm=zeros(iterations,10);
uopt2_admm=zeros(iterations,10);
uopt3_admm=zeros(iterations,10);
uopt4_admm=zeros(iterations,10);
u=zeros(4,iterations);

%% ADMM algorithm with quadprog
for k=1:iterations    
    
    uopt1=quadprog(2*(H1+rho/2*Aeq1'*Aeq1),(g1+y1(:,k)'*Aeq1+rho*(beq1-xf(:,k))'*Aeq1),Ain, bin,[],[],[],[],[],opts);
    uopt1_admm(k,:)=uopt1;
    
    uopt2=quadprog(2*(H2+rho/2*Aeq2'*Aeq2),(g2+y2(:,k)'*Aeq2+rho*(beq2-xf(:,k))'*Aeq2),Ain, bin,[],[],[],[],[],opts);
    uopt2_admm(k,:)=uopt2;
    
    uopt3=quadprog(2*(H3+rho/2*Aeq3'*Aeq3),(g3+y3(:,k)'*Aeq3+rho*(beq3-xf(:,k))'*Aeq3),Ain, bin,[],[],[],[],[],opts);
    uopt3_admm(k,:)=uopt3;
    
    uopt4=quadprog(2*(H4+rho/2*Aeq4'*Aeq4),(g4+y4(:,k)'*Aeq4+rho*(beq4-xf(:,k))'*Aeq4),Ain, bin,[],[],[],[],[],opts);
    uopt4_admm(k,:)=uopt4;
    
    xf(:,k+1)=1/N*((Aeq1*uopt1+beq1+1/rho*y1(:,k))+(Aeq2*uopt2+beq2+1/rho*y2(:,k))+(Aeq3*uopt3+beq3+1/rho*y3(:,k))+(Aeq4*uopt4+beq4+1/rho*y4(:,k)));
    
    y1(:,k+1)=y1(:,k)+rho*(Aeq1*uopt1+beq1-xf(:,k+1));
    y2(:,k+1)=y2(:,k)+rho*(Aeq2*uopt2+beq2-xf(:,k+1));
    y3(:,k+1)=y3(:,k)+rho*(Aeq3*uopt3+beq3-xf(:,k+1));
    y4(:,k+1)=y4(:,k)+rho*(Aeq4*uopt4+beq4-xf(:,k+1));
end

%% Error sequence plot
e_xf=0;
for i=1:size(xf,2)-1;
    e_xf(i)=norm(xf(:,i)-xfc);
end
figure(3)
loglog(e_xf,'Linewidth',1.1)
grid on
ylim([0.0001  10])
xlabel('Number of iterations','Interpreter','latex')
ylabel('Error','Interpreter','latex')
legend('$||x_f^{k}-x_f^{Central}||$ with $\rho=1$','Interpreter','latex')
%% Trajectories ADMM

u1=uopt1;
x1=[LTI1.x0; predmod1.T*LTI1.x0+predmod1.S*u1];
x11=x1(1:4:end);
x12=x1(2:4:end);
x13=x1(3:4:end);
x14=x1(4:4:end);
u2=uopt2;
x2=[LTI2.x0;predmod2.T*LTI2.x0+predmod2.S*u2];
x21=x2(1:4:end);
x22=x2(2:4:end);
x23=x2(3:4:end);
x24=x2(4:4:end);
u3=uopt3;
x3=[LTI3.x0;predmod3.T*LTI3.x0+predmod3.S*u3];
x31=x3(1:4:end);
x32=x3(2:4:end);
x33=x3(3:4:end);
x34=x3(4:4:end);
u4=uopt4;
x4=[LTI4.x0; predmod4.T*LTI4.x0+predmod4.S*u4];
x41=x4(1:4:end);
x42=x4(2:4:end);
x43=x4(3:4:end);
x44=x4(4:4:end);

for i=1:4
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

%% Plot results 

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

%% Various values of rho

r=[5 2 1 0.5 0.2];
e_xf=[];
iterations=100;
xf=zeros(N,iterations);
y1=zeros(4, iterations);
y2=zeros(4, iterations);
y3=zeros(4, iterations);
y4=zeros(4, iterations);

uopt1_admm=zeros(iterations,10);
uopt2_admm=zeros(iterations,10);
uopt3_admm=zeros(iterations,10);
uopt4_admm=zeros(iterations,10);
u=zeros(4,iterations);

for j=1:5
    
    rho=r(j);
    for k=1:iterations
        
        uopt1=quadprog(2*(H1+rho/2*Aeq1'*Aeq1),(g1+y1(:,k)'*Aeq1+rho*(beq1-xf(:,k))'*Aeq1),Ain, bin,[],[],[],[],[],opts);
        uopt1_admm(k,:)=uopt1;
        
        uopt2=quadprog(2*(H2+rho/2*Aeq2'*Aeq2),(g2+y2(:,k)'*Aeq2+rho*(beq2-xf(:,k))'*Aeq2),Ain, bin,[],[],[],[],[],opts);
        uopt2_admm(k,:)=uopt2;
        
        uopt3=quadprog(2*(H3+rho/2*Aeq3'*Aeq3),(g3+y3(:,k)'*Aeq3+rho*(beq3-xf(:,k))'*Aeq3),Ain, bin,[],[],[],[],[],opts);
        uopt3_admm(k,:)=uopt3;
        
        uopt4=quadprog(2*(H4+rho/2*Aeq4'*Aeq4),(g4+y4(:,k)'*Aeq4+rho*(beq4-xf(:,k))'*Aeq4),Ain, bin,[],[],[],[],[],opts);
        uopt4_admm(k,:)=uopt4;
       
        xf(:,k+1)=1/N*((Aeq1*uopt1+beq1+1/rho*y1(:,k))+(Aeq2*uopt2+beq2+1/rho*y2(:,k))+(Aeq3*uopt3+beq3+1/rho*y3(:,k))+(Aeq4*uopt4+beq4+1/rho*y4(:,k)));
        
        y1(:,k+1)=y1(:,k)+rho*(Aeq1*uopt1+beq1-xf(:,k+1));
        y2(:,k+1)=y2(:,k)+rho*(Aeq2*uopt2+beq2-xf(:,k+1));
        y3(:,k+1)=y3(:,k)+rho*(Aeq3*uopt3+beq3-xf(:,k+1));
        y4(:,k+1)=y4(:,k)+rho*(Aeq4*uopt4+beq4-xf(:,k+1));
        
    end    
    
    for i=1:size(xf,2)-1
        e_xf(i,j)=norm(xf(:,i)-xfc);
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



