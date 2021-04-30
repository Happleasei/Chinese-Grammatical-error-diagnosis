# encoding=utf8
# '''
# # 参数设置如下
# load
# bert
# embedding
# bert_config
# =
# modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
# model
# =
# modeling.BertModel(
# config=bert_config,
# is_training=True,
# input_ids=self.input_ids,
# input_mask=self.input_mask,
# token_type_ids=self.segment_ids,
# use_one_hot_embeddings=False)
# embedding
# =
# model.get_sequence_output()
# '''
import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from BERT_BILSTM_CRF.model import Model
from BERT_BILSTM_CRF.loader import load_sentences, update_tag_scheme
from BERT_BILSTM_CRF.loader import char_mapping, tag_mapping
from BERT_BILSTM_CRF.loader import augment_with_pretrained, prepare_dataset
from BERT_BILSTM_CRF.utils import get_logger, make_path, clean, create_model, save_model
from BERT_BILSTM_CRF.utils import print_config, save_config, load_config, test_ner
from BERT_BILSTM_CRF.data_utils import create_input, BatchManager
import os
flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
flags.DEFINE_integer("batch_size",  128,         "batch size")       # 128
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    200,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iob",      "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")  # 0.001
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_seq_len", 128,        "max sequence length for bert")
flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("train_file",   os.path.join("../Data", "train.char.bmes"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("../Data", "dev.char.bmes"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("../Data", "test.char.bmes"),   "Path for test data")

import os
FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(tag_to_id):
    config = OrderedDict()
    config["num_tags"] = len(tag_to_id)
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config['max_seq_len'] = FLAGS.max_seq_len

    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1

def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    #update_tag_scheme(train_sentences, FLAGS.tag_schema)
    #update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, FLAGS.batch_size)
    test_manager = BatchManager(test_data, FLAGS.batch_size)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)

                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger, global_steps=step)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def main(_):
    FLAGS.train = True
    FLAGS.clean = True
    clean(FLAGS)
    train()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)