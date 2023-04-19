import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy

import onnx
import onnxsim
import tensorflow as tf
from onnx_tf.backend import prepare
import tflite_runtime.interpreter as tflite

from config import config
from objects.model_inference import SingleNet as BasedPartyNet, InputNet
from generating_dataset import load_relevant_data_subset

import warnings
warnings.filterwarnings('ignore')

# Paths

sample_input = torch.rand(list(config.model.converter_sample_input_shape))
onnx_feat_gen_path = "./inference_artifacts/feature_gen.onnx"
tf_feat_gen_path = "./inference_artifacts/tf_feat_gen"
tf_model_path = "./inference_artifacts/tf_model_N"
onnx_model_path = "./inference_artifacts/asl_model_N.onnx"
tflite_model_path = "model.tflite"
tf_infer_model_path = "./inference_artifacts/tf_infer_model"
submission_path = "submission.zip"

# Feature converter

feature_converter = InputNet()


@nobuco.converter(F.normalize, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def normalize(input):
    return lambda x: tf.linalg.l2_normalize(x, axis=-1)


@nobuco.converter(torch.nn.modules.Linear, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def linear(a, b):
    return lambda x: x @ a.weight.detach().numpy().T + a.bias.detach().numpy()

# Initialize models

keras_models = []
for fold in config.split.folds_to_submit:
    model_infe = BasedPartyNet(fold, **config.model.params)
    model_infe.load_state_dict(
        torch.load(f"{config.paths.path_to_checkpoints}/fold_{fold}/best.pt", map_location='cpu')["model"],
    )

    sample_input = torch.rand(list(config.model.model_sample_input_shape))

    model_infe.eval()

    keras_model = nobuco.pytorch_to_keras(
        model_infe,
        args=[sample_input],
        inputs_channel_order=ChannelOrder.TENSORFLOW,
        outputs_channel_order=ChannelOrder.TENSORFLOW,
    )
    keras_model.append(keras_model)


class ASLInferModel(tf.Module):
    """The inference model."""

    def __init__(self, models):
        super(ASLInferModel, self).__init__()
        self.weights = config.split.weights
        self.feature_gen = feature_converter
        self.models = models
        self.feature_gen.trainable = False
        for model in self.models: model.trainable = False

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")
        ]
    ) 
    def call(self, inputs):
        output_tensors = {}
        features = self.feature_gen(tf.cast(inputs, dtype=tf.float32))
        output_tensors["outputs"] = tf.reduce_sum([self.models[f](features
        )[0, :] for f in range(len(config.split.folds_to_submit))], axis=0)
        return output_tensors
        
# Convert the model

mytfmodel = ASLInferModel(keras_models)
tf.saved_model.save(
    mytfmodel,
    tf_infer_model_path,
    signatures={"serving_default": mytfmodel.call},
)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_infer_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save and test the model

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()

found_signatures = list(interpreter.get_signature_list().keys())

prediction_fn = interpreter.get_signature_runner("serving_default")

start = time.time()
for _ in tqdm(range(100)):
    output = prediction_fn(inputs=load_relevant_data_subset(config.paths.pq_path))
    sign = np.argmax(output["outputs"])

print(time.time() - start)
print(sign, output["outputs"].shape)
os.system(f'zip {submission_path} {tflite_model_path}')
print("Model's Size (MB):", os.path.getsize('submission.zip')/1e6)