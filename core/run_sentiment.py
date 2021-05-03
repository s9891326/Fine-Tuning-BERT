import time
import click
import sys
import os
import shutil

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from itertools import chain
from transformers import BertTokenizerFast
from official.nlp import optimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from absl import logging
from definitions import OLD_FOLDER
from utils.enum_helper import ModelType, SaveFormat
from core.model import OriginModel, TFHubModel, CustomModel
from core.convert_helper import TensorRT, TFLite

TF_PREPROCESS = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"


def split_dataset(df):
    train_set, val_set = train_test_split(df,
                                          stratify=df['label'],
                                          test_size=0.05,
                                          random_state=42)
    return train_set, val_set


def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

    Args:
      sentence_features: a list with the names of string-valued features.
      seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
      A Keras Model that can be called on a list or dict of string Tensors
      (with the order or names, resp., given by sentence_features) and
      returns a dict of tensors for input to BERT.
    """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(TF_PREPROCESS)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


def map_example_to_dict(input_ids, input_mask, segment_ids, label=None):
    return {
               "input_ids": input_ids,
               "input_mask": input_mask,
               "segment_ids": segment_ids
           }, label


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self):
        self.test_file_name = "test.tsv"
        self.training_file_name = "train.tsv"
        self.latest_test_file_name = "latest_test.tsv"

    @abstractmethod
    def get_train_examples(self, data_dir):
        """Gets a collection of `TF.Dataset`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir):
        """Gets a collection of `TF.Dataset`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir):
        """Gets a collection of `TF.Dataset`s for prediction."""
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
        return self._read_tsv(Path(data_dir, self.training_file_name).as_posix())

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(Path(data_dir, self.latest_test_file_name).as_posix())

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_tsv(Path(data_dir, self.test_file_name).as_posix())

    def get_labels(self) -> List[str]:
        """See base class."""
        return ["0", "1", "2"]

    def get_names(self) -> List[str]:
        """Set Pandas names"""
        return ["text", "label"]


@click.command()
@click.option('--data_dir', default=None, show_default=True,
              help='The input data dir. Should contain the .tsv files (or other data files) for the task.',
              type=str)
@click.option('--task_name', default=None, show_default=True,
              help='The name of the task to train.', prompt=True,
              type=str, required=True)
@click.option('--output_dir', default=None, show_default=True,
              help='The output directory(savedModel) where the model checkpoints will be written.',
              prompt=True, type=str, required=True)
@click.option('--deploy_dir', default=None, show_default=True,
              help='If direct save the savedModel,will too large. Because there default save the optimizer.'
                   'So we save to .h5 format, and convert to savedModel, that will not save optimizer in savedModel.'
                   'And used this variable to deploy to tensorFlow serving',
              type=str)
@click.option('--learning_rate', default=2e-5, show_default=True,
              help='The initial learning rate for Adam.', type=float)
@click.option('--num_epochs', default=2, show_default=True,
              help='Total number of training epochs to perform.', type=int)
@click.option('--dropout_rate', default=0.1, show_default=True,
              help='Discard some neuron in the network to prevent cooperation between feature. '
                   'E.g., 0.1 = 10% of neuron.',
              type=float)
@click.option('--batch_size', default=16, show_default=True,
              help='Total batch size for training.', type=int)
@click.option('--max_seq_length', default=128, show_default=True,
              help='The maximum total input sequence length after WordPiece tokenization. '
                   'Sequences longer than this will be truncated, and sequences shorter '
                   'than this will be padded.',
              type=int)
@click.option('--do_train', default=False, show_default=True,
              help='Whether to run training.', type=bool)
@click.option('--do_test', default=False, show_default=True,
              help='Whether to run test on the dev set.', type=bool)
@click.option('--do_inference', default=False, show_default=True,
              help='Whether to run the model in inference mode on the test set.',
              type=bool)
@click.option('--save_format', default=["savedmodel", "tensorRT"], show_default=True,
              help='Default is ["savedmodel", "tensorRT"]. '
                   'Input have "savedmodel" will save savedmodel with h5 convert to savedmodel, '
                   'when input have "tensorRT" and "savedmodel", will save converted savedmodel to tensorRT, '
                   'if just "tensorRT" will save origin savedmodel to tensorRT, '
                   'input have "tflite" will save tflite format, but that is not correct work.',
              multiple=True)
@click.option('--model_type', default="tf-hub", show_default=True,
              help='Which model type do you want to use.'
                   'Default is "tf-hub", can use "custom"、"tf-hub"、"origin"', type=ModelType)
@click.option('--model_name', default="bert-base-chinese", show_default=True,
              help='Please input "bert-base-chinese"、"ckiplab/bert-base-chinese"、'
                   '"ckiplab/albert-base-chinese"、 "ckiplab/albert-tiny-chinese",'
                   'to set pretrained model name.',
              type=str)
@click.option('--load_model_dir', default=None, show_default=True,
              help='If you want load old model(savedModel) to train the new model, '
                   'please set where to load the old model dir.',
              type=str)
@click.option('--use_dev_dataset', default=False, show_default=True,
              help='Whether to evaluation the other dataset when do_test times. '
                   'The dataset name is latest_test.tsv', type=bool)
@click.option('--mix_number', default=0, show_default=True,
              help="How many training dataset to mix? If not set, the old dataset"
                   "will be mixed according to the number of tags in the new dataset",
              type=int)
@click.pass_context
class FineTuningBERT:
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.tokenizer_name = "bert-base-chinese"
        self.model_name = kwargs["model_name"]
        logging.info(f"model name: {self.model_name}")

        # load processors
        processors = {
            "sentiment": SentimentProcessor
            # "spam": SpamProcessor
        }
        self.task_name = kwargs["task_name"].lower()
        if self.task_name not in processors:
            raise ValueError("Task not found: %s" % (self.task_name))
        self.processor = processors[self.task_name]()
        self.label_list = self.processor.get_labels()

        # set TFRecord file name
        self.output_dir = kwargs["output_dir"]
        tf.io.gfile.makedirs(self.output_dir)

        # set up parameter
        self.deploy_dir = kwargs["deploy_dir"]
        self.model_type = ModelType(kwargs["model_type"])
        self.max_seq_length = kwargs["max_seq_length"]
        self.batch_size = kwargs["batch_size"]
        self.data_dir = kwargs["data_dir"]
        self.load_model_dir = kwargs["load_model_dir"]
        self.dropout_rate = kwargs["dropout_rate"]
        self.learning_rate = kwargs["learning_rate"]
        self.num_epochs = kwargs["num_epochs"]
        self.use_dev_dataset = kwargs["use_dev_dataset"]
        self.mix_number = kwargs["mix_number"]
        self.save_format = list(kwargs["save_format"])
        self.is_load_model_dir = bool(self.load_model_dir)
        self.meta = {}

        # load bert tokenizer or bert preprocess model
        self.bert_preprocess_model = make_bert_preprocess_model(["text"])
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)

        if kwargs["do_train"]:
            self.train()
        if kwargs["do_test"]:
            self.test()
        if kwargs["do_inference"]:
            self.inference()

    def convert_checkpoint_to_savedModel(self, model_path: str):
        model_suffix = "_h5_savedmodel"
        model_save_path = self.deploy_dir + model_suffix
        model = self.select_model()
        model.load_weights(model_path)
        model.save(model_save_path)
        return model_save_path

    def convert_example_to_feature(self, text):
        return self.tokenizer.encode_plus(text,
                                          add_special_tokens=True,  # add [CLS], [SEP]
                                          max_length=self.max_seq_length,  # max length of the text that can go to BERT
                                          padding='max_length',  # add [PAD] tokens
                                          return_attention_mask=True  # add attention mask to not focus on pad tokens
                                          )

    def encode_examples(self, ds, is_training: bool = False, multiple: bool = True):
        """prepare list, so that we can build up final TensorFlow dataset from slices."""
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        if multiple:
            for index, row in ds.iterrows():
                text = str(row["text"]).lower()
                label = row["label"]
                bert_input = self.convert_example_to_feature(text)

                input_ids_list.append(bert_input['input_ids'])
                token_type_ids_list.append(bert_input['token_type_ids'])
                attention_mask_list.append(bert_input['attention_mask'])
                label_list.append([label])
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_ids_list, attention_mask_list, token_type_ids_list, label_list))
        else:
            """prepare str, so that we can build up TensorFlow dataset from tensors"""
            feature = self.convert_example_to_feature(ds)
            dataset = tf.data.Dataset.from_tensors(
                (feature["input_ids"], feature["token_type_ids"], feature["attention_mask"]))

        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(map_example_to_dict)
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def load_dataset_from_pd(self, ds, is_training: bool = False):
        if isinstance(ds, pd.DataFrame):
            """load from dataset"""
            dataset = tf.data.Dataset.from_tensor_slices({"text": ds["text"], "label": ds["label"]})
        else:
            """load from inference"""
            dataset = tf.data.Dataset.from_tensor_slices({"text": [ds]})

        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size)

        if isinstance(ds, pd.DataFrame):
            dataset = dataset.map(lambda ex: (self.bert_preprocess_model(ex), ex["label"]))
        else:
            dataset = dataset.map(lambda ex: self.bert_preprocess_model(ex))
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def get_optimizer(self, dataset_size: int):
        # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/solve_glue_tasks_using_bert_on_tpu.ipynb
        steps_per_epoch = dataset_size // self.batch_size
        num_train_steps = steps_per_epoch * self.num_epochs
        num_warmup_steps = num_train_steps // 10

        optimizer = optimization.create_optimizer(
            init_lr=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')
        return optimizer

    def train(self):
        """fine tuning bert"""
        st = time.time()
        logging.info("--------training model--------")

        model = self.select_model()

        train_examples = self.processor.get_train_examples(self.data_dir)
        test_examples = self.processor.get_test_examples(self.data_dir)

        # 發現val_data主要的功能在判斷early_stop的標準、測試當下的epochs數值如何
        # 但由於資料集本身就不夠完整，所以不考慮切割資料來進行驗證
        if self.is_load_model_dir:
            logging.info("--------use load model--------")
            from utils.create_load_dataset import load_dataset
            # 如果是使用載入模型進行後續的訓練方式的話，會自動進行新舊資料集的混合，來提高正確率
            self.ctx.invoke(load_dataset, mix_number=self.mix_number)

            new_test_file_path = Path(self.data_dir, self.processor.test_file_name)
            if not new_test_file_path.exists():
                shutil.copy(Path(OLD_FOLDER, self.processor.test_file_name), new_test_file_path)

            if self.model_type == ModelType.TF_HUB:
                ds_train_encoded = self.load_dataset_from_pd(ds=train_examples, is_training=True)
                ds_val_encoded = None
                ds_test_encoded = self.load_dataset_from_pd(ds=test_examples)
            else:
                ds_train_encoded = self.encode_examples(train_examples)
                ds_val_encoded = None
                ds_test_encoded = self.encode_examples(test_examples)
                # savedModel load model
                # model = tf.keras.models.load_model(self.load_model_dir)
                # checkpoint load model
                # model.load_weights(self.load_model_dir)

            optimizer = self.get_optimizer(len(train_examples))
            model = tf.keras.models.load_model(
                self.load_model_dir,
                custom_objects={'AdamWeightDecay': optimizer})
        else:
            logging.info("--------use new model--------")
            if self.model_type == ModelType.TF_HUB:
                train_data, val_data = split_dataset(train_examples)
                ds_train_encoded = self.load_dataset_from_pd(ds=train_data, is_training=True)
                ds_val_encoded = self.load_dataset_from_pd(ds=val_data)
                ds_test_encoded = self.load_dataset_from_pd(ds=test_examples)
            else:
                train_data, val_data = split_dataset(train_examples)
                ds_train_encoded = self.encode_examples(train_data)
                ds_val_encoded = self.encode_examples(val_data)
                ds_test_encoded = self.encode_examples(test_examples)

            # optimizer Adam recommended
            # optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-6, clipnorm=1, decay=0.01)
            optimizer = self.get_optimizer(len(train_examples))

            # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[metric])

        train_spec = ds_train_encoded.take(1).element_spec
        for spec in train_spec:
            if isinstance(spec, dict):
                for key, value in spec.items():
                    logging.info(f"train input {key} shape: {value.shape}")
            else:
                logging.info(f"train label shape : {spec.shape}")

        model.summary()

        # strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
        # logging.info(f'Number of devices: {strategy.num_replicas_in_sync}')
        # with strategy.scope():

        # 載入初始化模型
        # tf.keras.utils.plot_model(model)  # plot model structure and save to model.png

        # early_stop = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=3)

        steps_per_epoch = len(train_examples) // self.batch_size
        log_dir = f"tensorBoard_logs/{self.output_dir.split('/')[0]}"
        # tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=3)
        tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # fit model
        bert_history = model.fit(
            ds_train_encoded,
            validation_data=ds_val_encoded,
            batch_size=self.batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=self.num_epochs,
            # callbacks=[early_stop, tensorBoard])
            callbacks=[tensorBoard])

        # evaluate test dataset
        model.evaluate(ds_test_encoded)

        st_pred = time.time()
        y_pred = model.predict(ds_test_encoded)
        self.meta["pred_time"] = round(time.time() - st_pred, 3)

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = test_examples["label"].to_list()

        report = classification_report(y_true, y_pred, target_names=self.label_list, digits=4)
        logging.info(report)
        self.meta["test_report"] = report

        self.meta["total_time"] = round(time.time() - st, 3)
        logging.info(f"history : {bert_history.history}")

        # save summary info
        model_summary_path = Path(self.output_dir, "summary.txt")
        logging.info(f"model summary path: {model_summary_path}")
        with open(model_summary_path, "w") as f:
            f.write(self.summary(bert_history))

        model.save(self.output_dir)
        self.save_action(model=model)

    def select_model(self):
        # fixme: use Enum to replace that
        # origin、tf-hub、custom
        if self.model_type == ModelType.ORIGIN:
            model = OriginModel(
                model_name=self.model_name,
                label_list=self.label_list,
                dropout_rate=self.dropout_rate,
                task_name=self.task_name
            ).get_model()
        elif self.model_type == ModelType.TF_HUB:
            model = TFHubModel(
                model_name=self.model_name,
                label_list=self.label_list,
                dropout_rate=self.dropout_rate,
                task_name=self.task_name
            ).get_model()
        else:
            model = CustomModel(
                model_name=self.model_name,
                label_list=self.label_list,
                dropout_rate=self.dropout_rate,
                task_name=self.task_name
            ).get_model()
        return model

    def load_model(self, dataset_size: int, examples: pd.DataFrame = None):
        ds_test_encoded = None
        if self.model_type == ModelType.TF_HUB:
            if examples is not None:
                ds_test_encoded = self.load_dataset_from_pd(ds=examples)
            optimizer = self.get_optimizer(dataset_size)
            model = tf.keras.models.load_model(
                self.output_dir,
                custom_objects={'AdamWeightDecay': optimizer})
        else:
            if examples is not None:
                ds_test_encoded = self.encode_examples(examples)
            model = tf.keras.models.load_model(self.output_dir)
        return model, ds_test_encoded

    def save_action(self, model):
        model_save_path = self.output_dir
        if SaveFormat.SAVEDMODEL.value in self.save_format:
            # 因為直接用savedModel部屬的話，會有GPU Memory太大，預測速度變久
            # 所以改存h5，再只load_weights存savedModel，這樣就不會存到optimizer等資訊
            model_path = self.output_dir + ".h5"
            model.save_weights(model_path)
            model_save_path = self.convert_checkpoint_to_savedModel(model_path)
        if SaveFormat.TENSORRT.value in self.save_format:
            tensorRT = TensorRT(output_dir=self.output_dir, deploy_dir=self.deploy_dir)
            tensorRT.convert(model_save_path)
        if SaveFormat.TFLITE.value in self.save_format:
            tflite = TFLite(output_dir=self.output_dir)
            tflite.convert()
        # checkpoint = tf.train.Checkpoint(model=model)
        # checkpoint.save(self.output_dir)
        # model.save_pretrained(save_directory=self.output_dir)

    def test(self):
        """test model"""
        st = time.time()
        logging.info("--------test model--------")

        # model = TFBertForSequenceClassification.from_pretrained(
        #     self.output_dir,
        #     num_labels=len(self.label_list))

        test_examples = self.processor.get_test_examples(self.data_dir)
        model, ds_test_encoded = self.load_model(
            dataset_size=len(test_examples),
            examples=test_examples)
        model.summary()

        test_spec = ds_test_encoded.element_spec
        for spec in test_spec:
            if isinstance(spec, dict):
                for key, value in spec.items():
                    logging.info(f"test input {key} shape: {value.shape}")
            else:
                logging.info(f"test label shape : {spec.shape}")

        model.evaluate(ds_test_encoded)
        self.test_report(
            model=model, ds_encoded=ds_test_encoded,
            examples=test_examples, st=st, name="test"
        )

        if self.use_dev_dataset:
            dev_examples = self.processor.get_dev_examples(self.data_dir)
            ds_dev_encoded = self.encode_examples(dev_examples)
            model.evaluate(ds_dev_encoded)
            self.test_report(
                model=model, ds_encoded=ds_dev_encoded,
                examples=dev_examples, st=st, name="latest_test"
            )

    def test_report(self,
                    model,
                    ds_encoded,
                    examples: pd.DataFrame,
                    st: float = time.time(),
                    name: str = "test"):
        st_pred = time.time()
        y_pred = model.predict(ds_encoded)
        self.meta["pred_time"] = round(time.time() - st_pred, 3)

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = examples["label"].to_list()

        report = classification_report(y_true, y_pred, target_names=self.label_list, digits=4)
        logging.info(report)

        self.meta["test_report"] = report
        self.meta["total_time"] = round(time.time() - st, 3)

        # save test info
        test_path = Path(self.output_dir, f"{name}_report.txt").as_posix()
        with open(test_path, "w") as f:
            f.write(self.summary())
        logging.info(f"success create {test_path}")

        if self.model_type == ModelType.TF_HUB:
            texts_batch = [
                self.tokenizer.batch_decode(ds[0]["input_word_ids"], skip_special_tokens=True)
                for ds in iter(ds_encoded)
            ]  # shape is (110, 32), 32 is batch_size
        else:
            texts_batch = [
                self.tokenizer.batch_decode(ds[0]["input_ids"], skip_special_tokens=True)
                for ds in iter(ds_encoded)
            ]  # shape is (110, 32), 32 is batch_size

        texts_batch = chain.from_iterable(texts_batch)  # convert two dimension to one dimension
        texts_batch = [t.replace(" ", "") for t in texts_batch]
        logging.info(f"text batch: {len(texts_batch)}")

        output_file_path = Path(self.output_dir, f"{name}_result.tsv").as_posix()
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(f"text\ttrue\tpred\n")
            for num, text in enumerate(texts_batch):
                f.write(f"{text}\t{y_true[num]}\t{y_pred[num]}\n")
        logging.info(f"success create {output_file_path}")

    def inference(self):
        """predict model"""
        logging.info("--------predict model--------")

        model, _ = self.load_model(dataset_size=self.batch_size)
        model.summary()
        compare_table = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }

        def show_result(pred):
            return f"positive:{pred[0][0]}, negative:{pred[0][1]}, neutral:{pred[0][2]}"

        while True:
            query = input("--> [input content]:\n")
            if self.model_type == ModelType.TF_HUB:
                _encode = self.load_dataset_from_pd(ds=query)
            else:
                _encode = self.encode_examples(query, multiple=False)

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
        - learning_rate: {m.optimizer.lr}
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
        - report:      \n{self.meta["test_report"]}
        - pred_time:     {self.meta["pred_time"]}s
        - total_time:    {self.meta["total_time"]}s
                """
        else:
            return f"""
        - batch_size:    {self.batch_size}
        - max_seq_length:{self.max_seq_length}
        - data_dir:      {self.data_dir}
        - report:      \n{self.meta["test_report"]}
        - pred_time:     {self.meta["pred_time"]}s
        - total_time:    {self.meta["total_time"]}s
                """


if __name__ == '__main__':
    FineTuningBERT()
