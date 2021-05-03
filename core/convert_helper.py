import abc
from abc import ABC
from typing import List

import tensorflow as tf


class ConvertHelper(ABC):
    def __init__(self,
                 output_dir: str,
                 deploy_dir: str = None):
        self.output_dir = output_dir
        self.deploy_dir = deploy_dir

    @abc.abstractmethod
    def convert(self):
        pass


class TFLite(ConvertHelper):
    def __init__(self,
                 output_dir: str,
                 model_name: str = "model",
                 **kwargs):
        self.model_name = f"{model_name}.tflite"
        self.input_shape = kwargs.get("input_shape", [1, 128])
        super(TFLite, self).__init__(output_dir=output_dir)

    def convert(self):
        model = tf.saved_model.load(self.output_dir)
        concrete_func = model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concrete_func.inputs[0].set_shape(self.input_shape)
        concrete_func.inputs[1].set_shape(self.input_shape)
        concrete_func.inputs[2].set_shape(self.input_shape)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        with tf.io.gfile.GFile(self.model_name, 'wb') as f:
            f.write(tflite_model)

    def inference(self):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=self.model_name)

        input_ids = [[101, 2769, 2695, 872, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0]]
        token_type_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0]]
        attention_mask = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0]]

        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], input_ids)
        interpreter.set_tensor(input_details[1]['index'], token_type_ids)
        interpreter.set_tensor(input_details[2]['index'], attention_mask)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)  # [[0.02583123 0.8985386  0.0756302 ]]


class TensorRT(ConvertHelper):
    def __init__(self,
                 output_dir: str,
                 deploy_dir: str,
                 **kwargs):
        self.model_suffix = "_tensorRT"
        self.max_workspace_size_bytes = kwargs.get("max_workspace_size_bytes", 1 << 28)  # 2G
        self.maximum_cached_engines = kwargs.get("maximum_cached_engines", 100)
        self.minimum_segment_size = kwargs.get("minimum_segment_size", 15)
        super(TensorRT, self).__init__(output_dir=output_dir, deploy_dir=deploy_dir)

    def convert(self, input_saved_model_dir: str = None):
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        # TF-TRT isn’t yet supported for Windows.
        # For now, you’ll have to use Linux - Ubuntu 16.04/18.04 are probably best supported at the moment.

        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=self.max_workspace_size_bytes,
            maximum_cached_engines=self.maximum_cached_engines,
            minimum_segment_size=self.minimum_segment_size)

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_saved_model_dir=self.deploy_dir + self.model_suffix)
