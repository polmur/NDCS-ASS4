function predmod=predmodgen(LTI,dim)

%Prediction matrices generation
%This function computes the prediction matrices to be used in the
%optimization problem

%Prediction matrix from initial state
T=zeros(dim.nx*(dim.N),dim.nx);
for k=1:dim.N
    T((k-1)*dim.nx+1:(k)*dim.nx,:)=LTI.A^k;
end

%Prediction matrix from input
S=zeros(dim.nx*(dim.N),dim.nu*(dim.N));
for k=1:dim.N
    for i=0:k-1
        S((k-1)*dim.nx+1:k*dim.nx,i*dim.nu+1:(i+1)*dim.nu)=LTI.A^(k-1-i)*LTI.B;
    end
end

predmod.T=T;
predmod.S=S;
