

import numpy as np
import mindspore as ms
import torch

from msg_transformer_mindspore import MSGTransformer
ms.set_context(mode=ms.PYNATIVE_MODE)
t = ms.Tensor(np.ones([8, 3, 224, 224]), ms.float32)
win = MSGTransformer()
x = win(t)
print(x, x.shape)

# from msg_transformer import MSGTransformer
# t = torch.zeros(1, 3, 224, 224)
# win = MSGTransformer()
# x = win(t)
# print(x,x.shape)

