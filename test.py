from model import ELICModel
import torch
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from ckbd import Cheng2020AnchorwithCheckerboard

elic = ELICModel(N=192,M=192)
m = JointAutoregressiveHierarchicalPriors(192,192)
c = Cheng2020AnchorwithCheckerboard(192)

x = torch.rand(2,3,64*2,64*4)
o = m(x)
e = elic(x)
cc = c(x)

print(" mbt_2028                  | elic                      | checkerboard")
print(o['x_hat'].shape,"|", e['x_hat'].shape,"|", cc['x_hat'].shape)
print(o['likelihoods']['y'].shape,"|",e['likelihoods']['y'].shape,"|",cc['likelihoods']['y'].shape)
print(o['likelihoods']['z'].shape,"|",e['likelihoods']['z'].shape,"|",cc['likelihoods']['z'].shape)

# print(sum(p.numel() for p in elic.parameters()))
# print(sum(p.numel() for p in m.parameters()))
# print(sum(p.numel() for p in c.parameters()))


elic.update()
s = elic.compress(x)
m.update()
s1=m.compress(x)
c.update()
s2 = c.compress(x)

print("elic",len(s['strings'][0]),[len(i) for i in s['strings'][0]])
print("mbt",len(s1['strings'][0]),[len(i) for i in s1['strings'][0]])
print("chbd",len(s2['strings'][0]),[len(i) for i in s2['strings'][0]])

# xh = elic.decompress(s['strings'],s['shape'])
# xh1 = m.decompress(s1['strings'],s1['shape'])
# xh2 = c.decompress(s2['strings'],s2['shape'])

#print(s)