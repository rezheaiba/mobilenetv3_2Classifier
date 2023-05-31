"""
# @Time    : 2022/8/7 12:21
# @File    : onnx2caffe2.py
# @Author  : rezheaiba
# ../weight/MobileNetV2_level1_newest.onnx
"""
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

# 参考： https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
"""
此方法的转换可以在移动设备上运行，但是转换的效果和用Caffe2Backend包一样的
from caffe2.python.onnx.backend import Caffe2Backend
"""
batch_size = 1

dummy_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

model = onnx.load("../weight/MobileNetV2_level1_newest.onnx")
onnx.checker.check_model(model)
prepared_backend = backend.prepare(model)
rep = backend.prepare(model, device="CUDA:0")
output = rep.run(dummy_data)

W = {model.graph.input[0].name: dummy_data}

c2_out = rep.run(W)[0]
print(c2_out)

# np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)
# print("Exported model has been executed on Caffe2 backend, and the result looks good!")

c2_workspace = rep.workspace
c2_model = rep.predict_net

from caffe2.python.predictor import mobile_exporter

init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())