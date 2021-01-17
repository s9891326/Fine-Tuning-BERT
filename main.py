from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import logging
from transformers import BertTokenizerFast, TFBertModel, BertConfig

from sklearn.model_selection import train_test_split


# The comments in the dataset : toxic severe_toxic obscene threat insult identity_hate

# def check_data_labels():
#     """check data labels balance"""
#     value_sum = [
#         df.toxic.sum(),
#         df.severe_toxic.sum(),
#         df.obscene.sum(),
#         df.threat.sum(),
#         df.insult.sum(),
#         df.identity_hate.sum(),
#     ]
#
#     x = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#     plt.bar(x, value_sum)
#     plt.show()


# dataset = tf.data.Dataset.from_tensor_slices((df.text.values, df.labels.values))
#
# for text, label in dataset.take(5):
#     print(f"text: {text}, label: {label}")


# train_df, eval_df = train_test_split(df, test_size=0.2)

# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def split_dataset(df):
    train_set, val_set = train_test_split(df,
                                          stratify=df['labels'],
                                          test_size=0.1)
    return train_set, val_set


def map_example_to_dict(input_ids, input_mask, segment_ids, label=None):
    return {
               "input_ids": input_ids,
               "input_mask": input_mask,
               "segment_ids": segment_ids
           }, label


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""

    @abstractmethod
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        pass

    @abstractmethod
    def get_labels(self):
        """Gets the list of labels for this data set."""
        pass

    @abstractmethod
    def get_names(self):
        """Gets the list of names for set pandas names."""
        pass

    def _read_csv(self, input_file, sep=False):
        """Reads a tab separated value file."""
        return pd.read_csv(
            input_file, )
        # sep="\t" if sep else None,
        # header=None,
        # names=self.get_names(),
        # engine="python")


class ToxicProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_csv(str(Path(data_dir, "train.csv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_csv(str(Path(data_dir, "dev.csv")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_csv(str(Path(data_dir, "test.csv")))

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["0", "0", "0", "0", "0", "0"]

    def get_names(self) -> List[str]:
        """Set Pandas names"""
        return ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class FineTuningBert:
    def __init__(self):
        self.batch_size = 64
        self.max_seq_length = 768
        self.tokenizer_name = "bert-base-uncased"
        self.data_dir = "data"

        # load processors
        processors = {
            # "sentiment": SentimentProcessor
            "toxic": ToxicProcessor
            # "spam": SpamProcessor
        }
        self.task_name = "toxic"
        if self.task_name not in processors:
            raise ValueError("Task not found: %s" % (self.task_name))
        self.processor = processors[self.task_name]()

        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name, do_lower_case=True)

    def convert_example_to_feature(self, text):
        return self.tokenizer.encode_plus(text,
                                          add_special_tokens=True,  # add [CLS], [SEP]
                                          max_length=self.max_seq_length,  # max length of the text that can go to BERT
                                          padding='max_length',  # add [PAD] tokens
                                          return_attention_mask=True  # add attention mask to not focus on pad tokens
                                          )

    def encode_examples(self, ds, limit=-1):
        """prepare list, so that we can build up final TensorFlow dataset from slices."""
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            ds = ds.take(limit)

        for index, row in ds.iterrows():
            text = row["text"]
            label = row["labels"]
            bert_input = self.convert_example_to_feature(text)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list)) \
            .map(map_example_to_dict)

    def custom_model(self):
        """Build BERT model."""
        # Load transformers config and set output_hidden_states to False
        config = BertConfig.from_pretrained(self.tokenizer_name)
        config.output_hidden_states = False

        transformer_model = TFBertModel.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_name,
            config=config
        )

        # Load the MainLayer
        bert = transformer_model.layers[0]

        # Build your model input
        input_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                          name='input_ids', dtype='int32')
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                           name='input_mask', dtype='int32')
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,),
                                            name='segment_ids', dtype='int32')
        inputs = {
            'input_ids': input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }

        # Load the Transformers BERT model as a layer in a Keras model
        bert_model = bert(inputs)[1]
        pooled_output = tf.keras.layers.Dropout(
            config.hidden_dropout_prob,
            name='pooled_output'
        )(bert_model)
        output = tf.keras.layers.Dense(
            units=len(self.processor.get_labels()),
            activation="sigmoid",
            name="probabilities"
        )(pooled_output)

        # And combine it all in a model object
        model = tf.keras.models.Model(
            inputs=inputs,
            outputs=output,
            name="CustomModel"
        )

        model.summary()
        return model

    def train(self):
        train_df = self.processor.get_train_examples(self.data_dir)

        train_df["labels"] = list(zip(
            train_df.toxic.tolist(),
            train_df.severe_toxic.tolist(),
            train_df.obscene.tolist(),
            train_df.threat.tolist(),
            train_df.insult.tolist(),
            train_df.identity_hate.tolist()
        ))

        train_df["text"] = train_df["comment_text"].apply(lambda x: x.replace("\n", ""))

        # train_data, val_data = split_dataset(train_df)

        ds_train_encoded = self.encode_examples(train_df[:100]).shuffle(1000).batch(self.batch_size)

        train_spec = ds_train_encoded.take(1).element_spec
        for spec in train_spec:
            if isinstance(spec, dict):
                for key, value in spec.items():
                    logging.info(f"train input {key} shape: {value.shape}")
            else:
                logging.info(f"train label shape : {spec.shape}")

        # print(ds_train_encoded)

        model = self.custom_model()

        # optimizer Adam recommended
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-5,
            epsilon=1e-08,
            clipnorm=1)

        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric
        )

        history = model.fit(
            ds_train_encoded,
            epochs=10
        )
        print(history)

    def evaluate(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    bert = FineTuningBert()
    bert.train()
