import os
import torch
import numpy as np

import onnx
import onnxsim
import tensorflow as tf
from onnx_tf.backend import prepare
import tflite_runtime.interpreter as tflite

from config import config
from objects.model import SingleNet as BasedPartyNet
from generating_dataset import InputNet, load_relevant_data_subset, FeatureGen

import warnings
warnings.filterwarnings('ignore')

# Paths

sample_input = torch.rand(list(config.model.converter_sample_input_shape))
onnx_feat_gen_path = "./inference_artifacts/feature_gen.onnx"
tf_feat_gen_path = "./inference_artifacts/tf_feat_gen"
tf_model_path = "./inference_artifacts/tf_model_N"
onnx_model_path = "./inference_artifacts/asl_model_N.onnx"
tflite_model_path = "./inference_artifacts/model.tflite"
tf_infer_model_path = "./inference_artifacts/tf_infer_model"
submission_path = "./inference_artifacts/submission.zip"


# Feature converter

feature_converter = InputNet()
feature_converter.eval()

torch.onnx.export(
    feature_converter,  # PyTorch Model
    sample_input,  # Input tensor
    onnx_feat_gen_path,  # Output file (eg. 'output_model.onnx')
    export_params = True,         
    opset_version = 12, 
    do_constant_folding=True,      
    input_names =  ['inputs'],     
    output_names = ['outputs'],  
    dynamic_axes={
        'inputs': {0: 'length'},
    },
)

model = onnx.load(onnx_feat_gen_path)
onnx.checker.check_model(model)
model_simple, check = onnxsim.simplify(model)
onnx.save(model_simple, onnx_feat_gen_path)

onnx_feat_gen = onnx.load(onnx_feat_gen_path)
tf_rep = prepare(onnx_feat_gen)
tf_rep.export_graph(tf_feat_gen_path)

# Initialize models

for fold in range(config.split.n_splits):
    model_infe = BasedPartyNet(**config.model.params)
    model_infe.load_state_dict(
        torch.load(f"{config.paths.path_to_checkpoints}/fold_{fold}/best.pt")["model"],
    )
    model_infe = model_infe.to(config.training.device)

    sample_input = torch.rand(list(config.model.model_sample_input_shape)).to(config.training.device)

    model_infe.eval()

    torch.onnx.export(
        model_infe,  # PyTorch Model
        sample_input,  # Input tensor
        onnx_model_path.replace('N', str(fold)),  # Output file (eg. 'output_model.onnx')
        export_params = True,         
        opset_version = 12, 
        do_constant_folding=True,      
        input_names =  ['inputs'],     
        output_names = ['outputs'],  
        dynamic_axes={
            'inputs': {0: 'length'},
        },
    )
    model = onnx.load(onnx_model_path.replace('N', str(fold)))
    onnx.checker.check_model(model)
    model_simple, check = onnxsim.simplify(model)
    onnx.save(model_simple, onnx_model_path.replace('N', str(fold)))

    onnx_model = onnx.load(onnx_model_path.replace('N', str(fold)))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path.replace('N', str(fold)))


class ASLInferModel(tf.Module):
    """The inference model."""

    def __init__(self):
        super(ASLInferModel, self).__init__()
        self.feature_gen = tf.saved_model.load(tf_feat_gen_path)
        self.models = [tf.saved_model.load(tf_model_path.replace('N', str(f))) for f in range(config.split.n_splits)]
        self.feature_gen.trainable = False
        for model in self.models: model.trainable = False

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 543, 2], dtype=tf.float32, name="inputs")
        ]
    )
    def call(self, inputs):
        output_tensors = {}
        features = self.feature_gen(**{"inputs": inputs})["outputs"]
        output_tensors["outputs"] = tf.reduce_mean([self.models[f](
            **{"inputs": features}
        )["outputs"][0, :] for f in range(config.split.n_splits)], axis=0)
        return output_tensors

# Convert the model

mytfmodel = ASLInferModel()
tf.saved_model.save(
    mytfmodel,
    tf_infer_model_path,
    signatures={"serving_default": mytfmodel.call},
)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_infer_model_path)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
tflite_model = converter.convert()

# Save and test the model

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()

found_signatures = list(interpreter.get_signature_list().keys())

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=load_relevant_data_subset(config.paths.pq_path))
sign = np.argmax(output["outputs"])

print(sign, output["outputs"].shape)

os.system(f'zip {submission_path} {tflite_model_path}')