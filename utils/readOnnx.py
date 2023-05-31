"""
# @Time    : 2022/8/8 18:26
# @File    : readOnnx.py
# @Author  : rezheaiba
"""
import onnx

# Load the ONNX model
model = onnx.load("../weight/MobileNetV2_level1_newest.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
