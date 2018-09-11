# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The ptb_data required for this example is in the ptb_data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/ptb_data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import reader
from word_LM_model import PTBModel
from word_LM_config import *
from my_tools.my_temp import format_print_han_list

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
# flags.DEFINE_string("data_path", 'ptb_data',
#                     "Where the training/test ptb_data is stored.")
flags.DEFINE_string("data_path", 'input/sogouNewsdata',
                    "Where the training/test ptb_data is stored.")
flags.DEFINE_string("save_path", 'log_root/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS


class PTBInput(object):
    """The input ptb_data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        # 每轮epoch需要的iteration数.也就是一轮epoch一共有 epoch_size 个 batch
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps

        # input_data 和 targets 的 shape 都是 [batch_size, num_steps]
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)
        # print(self.input_data.shape)
        # print(self.targets.shape)
        pass


def run_epoch_output(session, model, vocab, epoch_num, eval_op=None, verbose=False):
    """Runs the model on the given ptb_data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    # 先run 一下 initial_state ？
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "input_data": model.input_data,
        "output": model.output,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        # 将模型当前的状态 拿出来 放到 feed_dict中
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        # 取出输入和输出
        input = vals["input_data"]
        output = vals["output"]

        # 将input和output中的Id 映射回word
        output = output.flatten()
        input = input.flatten()
        input = [vocab.IdToWord(x) for x in input]
        output = [vocab.IdToWord(x) for x in output]
        # 写入output文件夹
        with open(os.path.join('output', 'temp_output.txt'), 'a', encoding='utf8') as fout:
            o1, o2 = format_print_han_list(input, output, space_num=10)
            fout.write('[epoch %d] input :'%epoch_num + ' '.join(o1) + '\n')
            fout.write('[epoch %d] output:'%epoch_num + ' '.join(o2) + '\n')
        # print('write to temp_output.txt')

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size,
                   np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given ptb_data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    # 先run一下initial_state ？
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        # 将模型当前的状态 拿出来 放到 feed_dict中
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size,
                   np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0":
        config.rnn_mode = BASIC
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB ptb_data directory")

    # 找到所有GPU设备
    # gpus = [
    #     x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
    # ]
    # if FLAGS.num_gpus > len(gpus):
    #     raise ValueError(
    #         "Your machine has only %d gpus "
    #         "which is less than the requested --num_gpus=%d."
    #         % (len(gpus), FLAGS.num_gpus))

    # 读取预处理好的数据
    # shape :
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, vocab = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 10

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            # reuse ?
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        # for name, model in models.items():
        #     model.export_ops(name)
        # metagraph = tf.train.export_meta_graph()
        # if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
        #     raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
        #                      "below 1.1.0")

        soft_placement = False
        # if FLAGS.num_gpus > 1:
        #     soft_placement = True
        #     util.auto_parallel(metagraph, m)

    # with tf.Graph().as_default():
    #     tf.train.import_meta_graph(metagraph)
    #     for model in models.values():
    #         model.import_ops()

        print('FLAGS.save_path ' + str(FLAGS.save_path))
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        config_proto.gpu_options.allow_growth = True
        # 成功使用自带的summary???
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                # 当epoch到达 config.max_epoch 次后，lr_decay 将会慢慢变小 （小于1的数乘以自己，越乘越小）
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                # lr_decay 越小， 当前的 learning_rate 也会越小
                m.assign_lr(session, config.learning_rate * lr_decay)

                for var in tf.trainable_variables():
                    print(var)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                test_perplexity = run_epoch_output(session, mtest, vocab, i)
                print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % 'last_'+FLAGS.save_path)
                sv.saver.save(session, 'last_'+FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
