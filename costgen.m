function [H,h]=costgen(predmod,dim,LTI)

H = predmod.S'*predmod.S + eye(dim.N*dim.nu);
h = 2*(predmod.S'*predmod.T*LTI.x0)';
 
end