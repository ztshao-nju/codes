import torch
import torch.nn as nn


a = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
b = a.detach()  # detach 后 requires_grad=False
b[0] = 100
print(a, b)  # 浅拷贝 还是会影响彼此 tensor([100.,   2.], requires_grad=True) tensor([100.,   2.])
a.detach_()
print(a, b)  # tensor([100.,   2.]) tensor([100.,   2.])
print('---------------------------------------------------')


x1 = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
x2 = torch.tensor([3, 4], dtype=torch.float, requires_grad=True)
x3 = torch.tensor([5, 6], dtype=torch.float, requires_grad=True)
y = (torch.pow(x1, 3) + torch.pow(x2, 2) + x3).sum()
y.backward()
x4 = x2.clone()
print(x1.grad, x2.grad, x3.grad, x4.grad, x4.requires_grad)
# 梯度分别是 3x^2, 2x, 1, None
# tensor([ 3., 12.]) tensor([6., 8.]) tensor([1., 1.]) None
print('---------------------------------------------------')




x3 = x1.clone()  # requires_grad为True 但是和x1有clone的梯度关系
x4 = x1.detach().clone()  # requires_grad为True 和x1无关
x5 = x1.detach().clone()
print(x3, x3.grad, x3.requires_grad)  # tensor([1., 2.], grad_fn=<CloneBackward>) None True
print(x4, x4.grad, x4.requires_grad)  # tensor([1., 2.]) None False

print('---------------------------------------------------')
x2.grad.zero_()  # 清空梯度
x5.requires_grad = True
z = (torch.pow(x1, 3) + torch.pow(x2, 2) + x3 + x4 + x5).sum()
z.backward()
print(x1.grad, x2.grad, x3.grad, x4.grad, x5.grad)
# tensor([ 7., 25.]) tensor([6., 8.]) None None tensor([1., 1.])
# x1梯度是原来梯度的两倍+1，因为累加了一次自己的梯度，然后加了一次clone后的x3的梯度 = 2x3+1 = 7
# x2梯度清空了 所以只有z求导后的梯度
# x3是非叶子变量 虽然requires_grad为True但是无梯度 是None
# x4是detach后的clone, requires_grad=False
# x5虽然是detach后的clone, 但是重新设置了requires_grad=True 所以可以正常求梯度
print(x1.requires_grad, x2.requires_grad, x3.requires_grad, x4.requires_grad, x5.requires_grad)
# True True True False True
