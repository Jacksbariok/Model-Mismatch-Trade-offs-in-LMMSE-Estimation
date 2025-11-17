clc
clear all

M = 100; % the number of simiulations
p = 30; % the dimension of variable x
% n = 5;   % the number of samples 
test = [5,15,25,29,30]; % the dimension of select variables

x = 1:1:50;
% x = [1,x];
c = 1;
for p_s = test
    num = 1;
    for i = x
        [mse(num,c),op(num)] = LMMSE(i,p_s,M,p);
        num = num + 1;
    end
    c = c+1;
end
figure
semilogy(x,op,'Color','k','Marker','square','LineWidth',2)
hold on
semilogy(x,mse(:,1),'Color','r','Marker','o','LineWidth',2)
hold on
semilogy(x,mse(:,2),'Marker','+','LineWidth',2)
hold on
semilogy(x,mse(:,3),'Marker','*','LineWidth',2)
hold on
semilogy(x,mse(:,4),'Marker','x','LineWidth',2)
hold on
semilogy(x,mse(:,5),'Marker','x','LineWidth',2)


function[final_ans,MMSE_o] = LMMSE(n,p_s,M,p)
mu_x = zeros(p,1);
sigma_x = 1;
K_x = sigma_x^2*eye(p);
K_hat_x = eye(p_s);
sigma_v = 0.5;
sigma_hat_z = 0;
mu_v = zeros(n,1);
K_v = sigma_v^2*eye(n);
K_hat_z = sigma_hat_z^2*eye(n);
for m = 1:M
    x = mvnrnd(mu_x,K_x)';
    v = mvnrnd(mu_v,K_v)';
    for j = 1:n
        A(j,:) = mvnrnd(mu_x,eye(p));
    end
    y = A*x + v;
    A_s = A(:,1:p_s);
    x_hat = K_hat_x*A_s'*pinv(A_s*K_hat_x*A_s'+K_hat_z)*y;
    x_true = K_x'*A'*pinv(A*K_x*A'+K_v)*y;
    J(m,1) = norm(x(1:p_s)-x_hat,2)^2 + sigma_x^2*(p-p_s) ;
    J_o(m,1) = norm(x-x_true,2)^2;
end
final_ans = sum(J)/M;
MMSE_o = sum(J_o)/M;
end
