# Adversarial-Example-Attack-and-Defense
- This repository contains the PyTorch implementation of the three non-target adversarial example attacks
- Contains these attacks implemented on MNIST and Time series datasets

## Attack
1. Fast Gradient Sign Method(FGSM) - [Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.](https://arxiv.org/abs/1412.6572)
```python
def fgsm_attack(input,epsilon,data_grad):
  pert_out = input + epsilon*data_grad.sign()
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out
```
2. Iterative Fast Gradient Sign Method(I-FGSM) - [A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016.](https://arxiv.org/abs/1607.02533)
```python
def ifgsm_attack(input,epsilon,data_grad):
  iter = 10
  alpha = epsilon/iter
  pert_out = input
  for i in range(iter-1):
    pert_out = pert_out + alpha*data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out
```
3. Momentum Iterative Fast Gradient Sign Method(MI-FGSM) - [Y. Dong et al. Boosting Adversarial Attacks with Momentum. arXiv preprint arXiv:1710.06081, 2018.](https://arxiv.org/abs/1710.06081)
```python
def mifgsm_attack(input,epsilon,data_grad):
  iter=10
  decay_factor=1.0
  pert_out = input
  alpha = epsilon/iter
  g=0
  for i in range(iter-1):
    g = decay_factor*g + data_grad/torch.norm(data_grad,p=1)
    pert_out = pert_out + alpha*torch.sign(g)
    pert_out = torch.clamp(pert_out, 0, 1)
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out
```
## Instructions to run 
- Clone the repository
- Run the notebooks present in the repository
  - To run the attack on MNIST dataset, run the `fgsm_mnist.ipynb` notebook
  - To run the attack on Time series dataset, run the `fgsm_ts.ipynb` notebook 