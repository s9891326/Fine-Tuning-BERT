import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from itertools import chain
from absl import app
from absl import flags
from absl import logging
from transformers import BertTokenizerFast, TFAutoModel
from transformers import TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import tensorflow as tf

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_float(
    "learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_epochs", 3,
    "Total number of training epochs to perform.")

flags.DEFINE_float(
    "dropout_rate", 0.1,
    "Discard some neuron in the network to prevent cooperation between feature."
    "E.g., 0.1 = 10% of neuron.")

flags.DEFINE_integer(
    "batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_train", False, "Whether to run training.")

flags.DEFINE_bool(
    "do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_demo", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "is_custom", True,
    "Whether to use custom model structure on the training.")

flags.DEFINE_string(
    "model_name", 'bert-base-chinese',
    "Please input 'bert-base-chinese'、'ckiplab/bert-base-chinese'、'ckiplab/albert-base-chinese'、"
    "'ckiplab/albert-tiny-chinese',to set pretrained model name.")

FLAGS = flags.FLAGS


def split_dataset(df):
    train_set, val_set = train_test_split(df,
                                          stratify=df['label'],
                                          test_size=0.05,
                                          random_state=42)
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

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        return pd.read_csv(input_file, sep="\t", header=None, names=self.get_names())


class SentimentProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(str(Path(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(str(Path(data_dir, "dev.tsv")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(str(Path(data_dir, "test.tsv")))

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["0", "1", "2"]

    def get_names(self) -> List[str]:
        """Set Pandas names"""
        return ["text", "label"]


# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, from_pt_word, model_name, label_list):
#         self.input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
#                                                     name="input_ids")
#         self.input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
#                                                 name="input_mask")
#         self.segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
#                                                  name="segment_ids")
#         self.model = TFAutoModel.from_pretrained(
#             pretrained_model_name_or_path=model_name,
#             from_pt=True if from_pt_word in model_name else False,
#             num_labels=len(label_list)
#         )
#         self.model._saved_model_inputs_spec = None
#         self.model.s
#
#
#
# class CustomModel(tf.keras.Model):
#     def __init__(self, from_pt_word, model_name, label_list):
#         super(CustomModel, self).__init__()
#         # self.encoder = EncoderLayer(from_pt_word, model_name, label_list)
#         # self.decoder =
#         self.model = TFAutoModel.from_pretrained(
#             pretrained_model_name_or_path=model_name,
#             from_pt=True if from_pt_word in model_name else False,
#             num_labels=len(label_list)
#         )
#         self.model._saved_model_inputs_spec = None
#         self.label_list = label_list
#
#     def call(self, input_word_ids, input_mask, segment_ids, training=False):
#         sequence_output = self.model([input_word_ids, input_mask, segment_ids])
#         out = tf.keras.layers.Dropout(FLAGS.dropout_rate)(sequence_output.pooler_output)
#         return tf.keras.layers.Dense(
#             units=len(self.label_list),
#             activation="softmax",
#             name="probabilities"
#         )(out)


class FineTuningBERT:
    def __init__(self):
        self.tokenizer_name = "bert-base-chinese"
        self.model_name = FLAGS.model_name
        logging.info(f"model name: {self.model_name}")

        # load processors
        processors = {
            "sentiment": SentimentProcessor
            # "spam": SpamProcessor
        }
        self.task_name = FLAGS.task_name.lower()
        if self.task_name not in processors:
            raise ValueError("Task not found: %s" % (self.task_name))
        self.processor = processors[self.task_name]()
        self.label_list = self.processor.get_labels()

        # load bert tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)

        # set TFRecord file name
        self.output_dir = FLAGS.output_dir
        tf.io.gfile.makedirs(self.output_dir)

        # set up parameter
        self.from_pt_word = "ckiplab/"
        self.is_custom = FLAGS.is_custom
        self.max_seq_length = FLAGS.max_seq_length
        self.batch_size = FLAGS.batch_size
        self.data_dir = FLAGS.data_dir
        self.meta = {}

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
            label = row["label"]
            bert_input = self.convert_example_to_feature(text)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])
        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, label_list))\
            .map(map_example_to_dict)

    def encode_example(self, text):
        """prepare str, so that we can build up TensorFlow dataset from tensors"""
        feature = self.convert_example_to_feature(text)
        return tf.data.Dataset.from_tensors(
            (feature["input_ids"], feature["token_type_ids"], feature["attention_mask"])).map(map_example_to_dict)

    def custom_model(self):
        """Build BERT model. Custom Input Layer(input_ids、input_mask、segment_ids) and Output Layer(Dropout、Dense)"""
        input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                               name="input_ids")
        input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                            name="segment_ids")

        # because ckiplab/models is use Pytorch develop so need to add 'from_pt' args
        model = TFAutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            from_pt=True if self.from_pt_word in self.model_name else False,
            num_labels=len(self.label_list)
        )
        model._saved_model_inputs_spec = None
        sequence_output = model([input_word_ids, input_mask, segment_ids])
        out = tf.keras.layers.Dropout(FLAGS.dropout_rate)(sequence_output.pooler_output)
        out = tf.keras.layers.Dense(
            units=len(self.label_list),
            activation="softmax",
            name="probabilities"
        )(out)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids],
            outputs=out,
            name=self.task_name)

        return model

    def origin_model(self):
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

    def train(self):
        """fine tuning bert"""
        st = time.time()
        logging.info("--------training model--------")
        train_examples = self.processor.get_train_examples(self.data_dir)
        test_examples = self.processor.get_test_examples(self.data_dir)

        train_data, val_data = split_dataset(train_examples)
        ds_train_encoded = self.encode_examples(train_data).shuffle(100).batch(self.batch_size)
        ds_val_encoded = self.encode_examples(val_data).batch(self.batch_size)
        ds_test_encoded = self.encode_examples(test_examples).batch(self.batch_size)

        train_spec = ds_train_encoded.take(1).element_spec
        for spec in train_spec:
            if isinstance(spec, dict):
                for key, value in spec.items():
                    logging.info(f"train input {key} shape: {value.shape}")
            else:
                logging.info(f"train label shape : {spec.shape}")

        # strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
        # logging.info(f'Number of devices: {strategy.num_replicas_in_sync}')
        # with strategy.scope():
        # input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
        #                                        name="input_ids")
        # input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
        #                                    name="input_mask")
        # segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
        #                                     name="segment_ids")
        # model = CustomModel(self.from_pt_word, self.model_name, self.label_list)
        # y = model(input_word_ids, input_mask, segment_ids)
        # print(y)

        model = self.custom_model() if self.is_custom else self.origin_model()
        model.summary()

        # optimizer Adam recommended
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=1e-08, clipnorm=1)

        # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3)

        log_dir = f"logs/{self.model_name}"
        tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric])

        # fit model
        bert_history = model.fit(
            ds_train_encoded,
            validation_data=ds_val_encoded,
            batch_size=self.batch_size,
            epochs=FLAGS.num_epochs,
            callbacks=[early_stop, tensorBoard])

        # evaluate test set
        model.evaluate(ds_test_encoded)

        st_pred = time.time()
        y_pred = model.predict(ds_test_encoded)
        self.meta["pred_time"] = round(time.time() - st_pred, 3)

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = test_examples["label"].to_list()

        report = classification_report(y_true, y_pred, target_names=self.label_list, digits=4)
        logging.info(report)
        self.meta["eval_report"] = report

        self.meta["total_time"] = round(time.time() - st, 3)
        logging.info(f"history : {bert_history.history}")

        # save summary info
        model_summary_path = Path(self.output_dir, "summary.txt")
        logging.info(f"model summary path: {model_summary_path}")
        with open(model_summary_path, "w") as f:
            f.write(self.summary(bert_history))

        # signatures = {
        #     'default': model.serve.get_concrete_function(),
        # }
        model.save(
            filepath=self.output_dir,
            overwrite=True,
            include_optimizer=True,
            save_format="tf"
        )
        # model.save_pretrained(save_directory=self.output_dir)

    def eval(self):
        """eval model"""
        st = time.time()
        logging.info("--------eval model--------")
        test_examples = self.processor.get_test_examples(self.data_dir)
        ds_test_encoded = self.encode_examples(test_examples).batch(self.batch_size)

        test_spec = ds_test_encoded.element_spec
        for spec in test_spec:
            if isinstance(spec, dict):
                for key, value in spec.items():
                    logging.info(f"test input {key} shape: {value.shape}")
            else:
                logging.info(f"test label shape : {spec.shape}")

        model = tf.keras.models.load_model(self.output_dir)
        model.summary()

        # model = TFBertForSequenceClassification.from_pretrained(
        #     self.output_dir,
        #     num_labels=len(self.label_list))

        st_pred = time.time()
        y_pred = model.predict(ds_test_encoded)
        self.meta["pred_time"] = round(time.time() - st_pred, 3)

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = test_examples["label"].to_list()

        report = classification_report(y_true, y_pred, target_names=self.label_list, digits=4)
        logging.info(report)

        self.meta["eval_report"] = report
        self.meta["total_time"] = round(time.time() - st, 3)

        # save eval info
        eval_path = Path(self.output_dir, "eval_report.txt")
        with open(eval_path, "w") as f:
            f.write(self.summary())
        logging.info("success create eval_report.txt")

        texts_batch = [
            self.tokenizer.batch_decode(ds[0]["input_ids"], skip_special_tokens=True)
            for ds in iter(ds_test_encoded)
        ]  # shape is (110, 32), 32 is batch_size
        texts_batch = chain.from_iterable(texts_batch)  # convert two dimension to one dimension
        texts_batch = [t.replace(" ", "") for t in texts_batch]
        logging.info(f"text batch: {len(texts_batch)}")

        output_test_file = str(Path(self.output_dir, "test_results.tsv"))
        with open(output_test_file, "w", encoding="utf-8") as f:
            f.write(f"text\ttrue\tpred\n")
            for num, text in enumerate(texts_batch):
                f.write(f"{text}\t{y_true[num]}\t{y_pred[num]}\n")
        logging.info("success create test_results.tsv")

    def predict(self):
        """predict model"""
        logging.info("--------predict model--------")

        model = tf.keras.models.load_model(self.output_dir)

        compare_table = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }

        def show_result(pred):
            return f"positive:{pred[0][0]}, negative:{pred[0][1]}, neutral:{pred[0][2]}"

        while True:
            query = input("--> [input content]:\n")
            _encode = self.encode_example(query).batch(self.batch_size)

            y_pred = model.predict(_encode)
            logging.info(show_result(y_pred))

            y_pred = tf.math.argmax(y_pred, axis=-1)
            logging.info(f"predict result : {compare_table[y_pred.numpy()[0]]}")

    def summary(self, history=None):
        if history:
            h = history.history
            m = history.model
            return f"""
        - epoch:         {history.epoch}
        - params:        {history.params}
        - learning_rate: {m.optimizer.lr.numpy()}
        - batch_size:    {self.batch_size}
        - max_seq_length:{self.max_seq_length}
        - data_dir:      {self.data_dir}
        - loss_name:     {m.loss.name}
        - loss:          {h["loss"]}
        - accuracy:      {h["accuracy"]}
        - val_loss:      {h.get("val_loss")}
        - val_accuracy:  {h.get("val_accuracy")}
        - input_names:   {m.input_names}
        - output_names:  {m.output_names}
        - output_shape:  {m.output_shape}
        - report:      \n{self.meta["eval_report"]}
        - pred_time:     {self.meta["pred_time"]}s
        - total_time:    {self.meta["total_time"]}s
                """
        else:
            return f"""
        - batch_size:    {self.batch_size}
        - max_seq_length:{self.max_seq_length}
        - data_dir:      {self.data_dir}
        - report:      \n{self.meta["eval_report"]}
        - pred_time:     {self.meta["pred_time"]}s
        - total_time:    {self.meta["total_time"]}s
                """


def main(_):
    _bert = FineTuningBERT()

    if FLAGS.do_train:
        _bert.train()
    if FLAGS.do_eval:
        _bert.eval()
    if FLAGS.do_demo:
        _bert.predict()


if __name__ == '__main__':
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
