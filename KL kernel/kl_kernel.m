function [K_train,K_test2train, K_train2test] = kl_kernel(K_train,K_test2train, K_train2test,sigma,bias)
K_train = exp(-sigma * K_train + bias);
K_test2train = exp(-sigma * K_test2train + bias);
K_train2test = exp(-sigma * K_train2test + bias);
end

