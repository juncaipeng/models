# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import os
import time
import sys

import paddle.fluid as fluid
import paddle

import utils
import reader
import creator
sys.path.append('../shared_modules/models/')
from model_check import check_cuda
from model_check import check_version

paddle.enable_static()

parser = argparse.ArgumentParser(__doc__)
# 1. model parameters
model_g = utils.ArgumentGroup(parser, "model", "model configuration")
model_g.add_arg("load_model_path", str, "./infer_model", "")
model_g.add_arg("model_filename", str, "", "")
model_g.add_arg("params_filename", str, "", "")
model_g.add_arg("word_emb_dim", int, 128,
                "The dimension in which a word is embedded.")
model_g.add_arg("grnn_hidden_dim", int, 128,
                "The number of hidden nodes in the GRNN layer.")
model_g.add_arg("bigru_num", int, 2,
                "The number of bi_gru layers in the network.")
model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

# 2. data parameters
data_g = utils.ArgumentGroup(parser, "data", "data paths")
data_g.add_arg("word_dict_path", str, "./conf/word.dic",
               "The path of the word dictionary.")
data_g.add_arg("label_dict_path", str, "./conf/tag.dic",
               "The path of the label dictionary.")
data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic",
               "The path of the word replacement Dictionary.")
data_g.add_arg("test_data", str, "./data/train.tsv",
               "The folder where the training data is located.")
data_g.add_arg("test_nums", int, 1000, "")


def do_eval(args):
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    [infer_program, feed_target_names, fetch_targets] = \
        fluid.io.load_inference_model(
            args.load_model_path,
            exe,
            model_filename=args.model_filename,
            params_filename=args.params_filename)

    dataset = reader.Dataset(args)
    data_reader = dataset.file_reader(args.test_data)

    idx = 0
    right_num = 0
    for data, label in data_reader():
        base_shape = [[len(data)]]
        lod = [np.array(data).astype(np.int64)]
        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
        res = exe.run(infer_program,
                      feed = {feed_target_names[0]: tensor_words},
                      fetch_list=fetch_targets,
                      return_numpy=False)
        idx += 1
        if all(np.array(res[0]).ravel() == np.array(label).ravel()):
            right_num += 1
        if args.test_nums > 0 and idx >= args.test_nums:
            break
    print("right ration:" + str(right_num / idx))

if __name__ == '__main__':
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    check_version()
    
    if args.model_filename == "":
        args.model_filename = None
    if args.params_filename == "":
        args.params_filename = None

    do_eval(args)
