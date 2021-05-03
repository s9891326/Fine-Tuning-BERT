import abc
from abc import ABC
from typing import List

from transformers import TFAutoModel
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import tensorflow_hub as hub


class Model(ABC):
    def __init__(self,
                 model_name: str,
                 label_list: List,
                 **kwargs):
        self.from_pt_word = "ckiplab/"
        self.model_name = model_name
        self.label_list = label_list
        self.dropout_rate = kwargs["dropout_rate"]
        self.task_name = kwargs["task_name"]

        self.tf_encoder_url = kwargs.get("tf_encoder_url", "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3")
        self.dType = kwargs.get("dType", tf.int32)
        self.layer_shape = kwargs.get("layer_shape", (None,))

    @abc.abstractmethod
    def get_model(self):
        pass

    def get_layers_input(self):
        """Output three inputs of first layers"""
        input_word_ids = tf.keras.layers.Input(shape=self.layer_shape, dtype=self.dType,
                                               name="input_ids")
        input_mask = tf.keras.layers.Input(shape=self.layer_shape, dtype=self.dType,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=self.layer_shape, dtype=self.dType,
                                            name="segment_ids")
        return input_word_ids, input_mask, segment_ids

    def get_fine_tuning_layer(self, pooler_output):
        """Output one layer of fine tuning layer"""
        out = tf.keras.layers.Dropout(self.dropout_rate)(pooler_output)
        out = tf.keras.layers.Dense(
            units=len(self.label_list),
            activation="softmax",
            name="probabilities"
        )(out)

        return out


class CustomModel(Model):
    def __init__(self,
                 model_name: str,
                 label_list: List,
                 **kwargs):
        super(CustomModel, self).__init__(
            model_name=model_name,
            label_list=label_list,
            **kwargs)

    def get_model(self):
        """Build BERT model. Custom Input Layer(input_ids、input_mask、segment_ids) and Output Layer(Dropout、Dense)"""
        input_word_ids, input_mask, segment_ids = super().get_layers_input()

        # because ckiplab/models is use Pytorch develop so need to add 'from_pt' args
        model = TFAutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            from_pt=True if self.from_pt_word in self.model_name else False,
            num_labels=len(self.label_list)
        )
        model._saved_model_inputs_spec = None
        sequence_output = model([input_word_ids, input_mask, segment_ids])
        out = super().get_fine_tuning_layer(pooler_output=sequence_output.pooler_output)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids],
            outputs=out,
            name=self.task_name)

        return model


class OriginModel(Model):
    def __init__(self,
                 model_name: str,
                 label_list: List,
                 **kwargs):
        super(OriginModel, self).__init__(
            model_name=model_name,
            label_list=label_list,
            **kwargs)

    def get_model(self):
        """
        Build BERT model. Using origin way to build BERT model On the Transformers package.

        But this way don't custom input and output, so that can't adapt to deepnlp project worker.
        Need to change deepnlp project parameter.
        """
        model = TFBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            from_pt=True if self.from_pt_word in self.model_name else False,
            num_labels=len(self.label_list)
        )

        return model


class TFHubModel(Model):
    def __init__(self,
                 model_name: str,
                 label_list: List,
                 **kwargs):
        super(TFHubModel, self).__init__(
            model_name=model_name,
            label_list=label_list,
            **kwargs)

    def get_model(self):
        input_word_ids, input_mask, input_type_ids = super().get_layers_input()

        encoder_inputs = dict(
            input_word_ids=input_word_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids
        )
        encoder = hub.KerasLayer(
            self.tf_encoder_url,
            trainable=True,
            name="encoder")
        outputs = encoder(encoder_inputs)["pooled_output"]
        out = super().get_fine_tuning_layer(pooler_output=outputs)

        return tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=out,
            name=self.task_name)
