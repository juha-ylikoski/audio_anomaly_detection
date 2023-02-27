from model import ELICModel
import torch
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from ckbd import Cheng2020AnchorwithCheckerboard
from torch.optim import Adam
from compressai.losses import RateDistortionLoss

elic = ELICModel()#N=20,M=20,block_sizes=[2,2,2,4,10])
#elic = JointAutoregressiveHierarchicalPriors(25,25)
#elic = Cheng2020AnchorwithCheckerboard(20)
print(sum(p.numel() for p in elic.parameters()))

optimizer = Adam(elic.parameters())
loss_fn = RateDistortionLoss()
epochs=200
x = torch.rand(2,3,64*2,64*2)
# o = m(x)
# e = elic(x)
# cc = c(x)

for e in range(epochs):
    optimizer.zero_grad()
    x_hat = elic(x)
    loss = loss_fn(x_hat, x)
    loss['loss'].backward()
    optimizer.step()
    if e%20==0:
        print("epoch:",e,end="|")
        print([f"{k}:{v.item():10.4f}" for k,v in loss.items()])
    


# print(" mbt_2028                  | elic                      | checkerboard")
# print(o['x_hat'].shape,"|", e['x_hat'].shape,"|", cc['x_hat'].shape)
# print(o['likelihoods']['y'].shape,"|",e['likelihoods']['y'].shape,"|",cc['likelihoods']['y'].shape)
# print(o['likelihoods']['z'].shape,"|",e['likelihoods']['z'].shape,"|",cc['likelihoods']['z'].shape)

# print(sum(p.numel() for p in m.parameters()))
# print(sum(p.numel() for p in c.parameters()))


elic.update()
s = elic.compress(x)
# m.update()
# s1=m.compress(x)
# c.update()
# s2 = c.compress(x)

print("elic",len(s['strings'][0]),[len(i) for i in s['strings'][0]])
# print("mbt",len(s1['strings'][0]),[len(i) for i in s1['strings'][0]])
# print("chbd",len(s2['strings'][0]),[len(i) for i in s2['strings'][0]])

xh = elic.decompress(s['strings'],s['shape'])
# xh1 = m.decompress(s1['strings'],s1['shape'])
# xh2 = c.decompress(s2['strings'],s2['shape'])

print(xh['x_hat'].shape)
print(torch.mean((xh['x_hat']-x)**2).item())
#print(s)