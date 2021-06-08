import torch
import struct
import sys
from utils.torch_utils import select_device

# Initialize
device = select_device('cpu')
pt_file = "yolov5s.pt" # sys.argv[1]
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

# print(model.state_dict())
print(list(model.state_dict().keys()))
print(model.names)

with open(pt_file.split('.')[0] + '.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

#    f.write('{}\n'.format(len(model.names)))
#    for name in model.names:
#        f.write(name)
#        f.write('\n')
