import torch
import torch.nn as nn
import onnx
import numpy as np
import onnx.numpy_helper

from onnx import numpy_helper as nh


def create_initializer_tensor(
		name: str,
		tensor_array: np.ndarray,
		data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
	# (TensorProto)
	initializer_tensor = onnx.helper.make_tensor(
		name=name,
		data_type=data_type,
		dims=tensor_array.shape,
		vals=tensor_array.flatten().tolist())

	return initializer_tensor


# upsample: Conv_7; Conv_82
# simp: Conv_7; Conv_13


def split_carvana(path):
	model_name = path

	onnx_model = onnx.load(model_name)

	if ("simp" in model_name):
		st_node, ed_node = "Conv_7", "Conv_13"
	elif ("upsample" in model_name):
		st_node, ed_node = "Conv_7", "Conv_82"
	else:
		raise NotImplementedError

	new_initializers = []
	new_nodes = []

	initializers = {}
	for onnx_module in onnx_model.graph.initializer:
		initializers[onnx_module.name] = nh.to_array(onnx_module)

	trigger = False
	for i, node in enumerate(onnx_model.graph.node):
		if (node.name == st_node): trigger = True
		if (trigger == True):
			for init_name in node.input:
				if (init_name not in initializers): continue
				conv_initializer = create_initializer_tensor(
					name=init_name,
					tensor_array=initializers[init_name],
					data_type=onnx.TensorProto.FLOAT
				)
				new_initializers.append(conv_initializer)
			new_nodes.append(node)
		if (node.name == ed_node): break

	model_input_name = new_nodes[0].input[0]
	X = onnx.helper.make_tensor_value_info(model_input_name,
										   onnx.TensorProto.FLOAT,
										   [None, 3, 31, 47])
	model_output_name = new_nodes[-1].output[0]
	Y = onnx.helper.make_tensor_value_info(model_output_name,
										   onnx.TensorProto.FLOAT,
										   [None, 2, 31, 47])

	new_graph = onnx.helper.make_graph(
		nodes=new_nodes,
		name="SplitNet",
		inputs=[X],
		outputs=[Y],
		initializer=onnx_model.graph.initializer
	)

	model_def = onnx.helper.make_model(new_graph, producer_name="onnx_example")

	model_def.opset_import[0].version = 12
	model_def = onnx.shape_inference.infer_shapes(model_def)

	onnx.checker.check_model(model_def)
	new_model_name = model_name[:-5] + "_split.onnx"
	onnx.save(model_def, new_model_name)
	print('New Carvana model saved to', new_model_name)
