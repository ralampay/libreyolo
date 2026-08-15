import torch

from libreyolo.models.yolo9.nn import Backbone9


model = Backbone9(config="s")
model.eval()

x = torch.randn(
    1,
    3,
    640,
    640,
)

with torch.no_grad():
    p3, p4, p5 = model(x)

print("P3:", p3.shape)
print("P4:", p4.shape)
print("P5:", p5.shape)
