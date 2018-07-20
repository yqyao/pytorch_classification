import torch
import sys

sys.path.append("../")
from models.model_builder import model_builder

state_dict = torch.load("../weights/vgg16_best_model.pth")



from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    if k.split(".")[1] == "fc":
        continue
    new_state_dict[name] = v

net = model_builder("vgg16", pretrained=False, num_classes=20)
net.load_state_dict(new_state_dict)
torch.save(net.state_dict(), "../weights/vgg16_best_feature.pth")
