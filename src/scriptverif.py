from rndiy import *
import numpy as np

np.random.seed(1)

datax = np.random.randn(20,10)
datay = np.random.choice([-1,1],20,replace=True)
dataymulti = np.random.choice(range(10),20,replace=True)
linear = Linear(10,1)
sigmoide = Sigmoide()
softmax = Softmax()
tanh = Tanh()
relu = ReLU()
conv1D = Conv1D(k_size=3,chan_in=1,chan_out=5,stride=2)
maxpool1D = MaxPool1D(k_size=2,stride=2)
flatten = Flatten()

mse = MSE()
bce = BCE()
crossentr = CrossEntropy() #cross entropy avec log softmax

## Lineaire et MSE
linear.zero_grad()
res_lin = linear.forward(datax)
res_mse = mse.forward(datay.reshape(-1,1), res_lin)
delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
linear.backward_update_gradient(datax,delta_mse)
grad_lin = linear._gradient
delta_lin = linear.backward_delta(datax,delta_mse)

## Tanh, Sigmoide, ReLU
res_tanh = tanh.forward(res_lin)
delta_tanh = tanh.backward_delta(res_lin, delta_mse)
res_sig = sigmoide.forward(res_lin)
delta_sig = sigmoide.backward_delta(res_lin, delta_mse)
res_relu= relu.forward(res_lin)
delta_relu = relu.backward_delta(res_lin, delta_mse)

## Softmax, BCE, CrossEntropy
soft_lin = softmax.forward(res_lin)
res_bce = bce.forward((datay>0).astype(float).reshape(-1,1),soft_lin)
delta_bce = bce.backward((datay>0).astype(float).reshape(-1,1),soft_lin)
res_ce = crossentr.forward(dataymulti,datax)
delta_ce = crossentr.backward(dataymulti,datax)

## Convolutions
linear.zero_grad()
conv1D.zero_grad()
res_conv = conv1D.forward(datax.reshape(20,10,1))
res_pool = maxpool1D.forward(res_conv)
res_flat = flatten.forward(res_pool)
res_linconv = linear.forward(res_flat)
res_mseconv = mse.forward(datay.reshape(-1,1),res_linconv)
delta_mseconv = mse.backward(datay.reshape(-1,1),res_linconv)
delta_linconv = linear.backward_delta(res_flat,delta_mseconv)
delta_flat = flatten.backward_delta(res_pool,delta_linconv)
delta_pool = maxpool1D.backward_delta(res_conv,delta_flat)
delta_conv= conv1D.backward_delta(datax.reshape(20,10,1),delta_pool)
conv1D.backward_update_gradient(datax.reshape(20,10,1),delta_pool)
grad_conv = conv1D._gradient


np.savez_compressed("verif_projet.npz",res_lin=res_lin,res_mse=res_mse,delta_mse=delta_mse,grad_lin=grad_lin,delta_lin=delta_lin,res_tanh=res_tanh,delta_tanh=delta_tanh,res_sig=res_sig,delta_sig=delta_sig,res_relu=res_relu,delta_relu=delta_relu,soft_lin=soft_lin,res_bce=res_bce,delta_bce=delta_bce,res_ce=res_ce,delta_ce=delta_ce,res_conv=res_conv,res_pool=res_pool,res_flat=res_flat,res_linconv=res_linconv,res_mseconv=res_mseconv,delta_mseconv=delta_mseconv,delta_linconv=delta_linconv,delta_flat=delta_flat,delta_pool=delta_pool,delta_conv=delta_conv,grad_conv=grad_conv)