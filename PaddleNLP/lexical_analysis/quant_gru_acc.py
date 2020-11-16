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

import argparse
import os
import time
import sys

import paddle
import paddle.fluid as fluid

import utils
import reader
import creator
sys.path.append('../shared_modules/models/')
from model_check import check_cuda
from model_check import check_version
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization

def parse_args(): 
    parser = argparse.ArgumentParser(__doc__)
    # 1. model parameters
    model_g = utils.ArgumentGroup(parser, "model", "model configuration")
    model_g.add_arg("model_load_path", str, "./gru_acc_model", "model path")
    model_g.add_arg("model_save_path", str, "./gru_acc_quant_model", "model path")
    model_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")

    # 2. data parameters
    data_g = utils.ArgumentGroup(parser, "data", "data paths")
    data_g.add_arg("word_dict_path", str, "./conf/word.dic",
                   "The path of the word dictionary.")
    data_g.add_arg("label_dict_path", str, "./conf/tag.dic",
                   "The path of the label dictionary.")
    data_g.add_arg("word_rep_dict_path", str, "./conf/q2b.dic",
                   "The path of the word replacement Dictionary.")
    data_g.add_arg("test_data", str, "./data/test.tsv",
                   "The folder where the training data is located.")
    data_g.add_arg("batch_size", int, 1, "")
    data_g.add_arg("batch_num", int, 1000, "")
    
    return parser.parse_args()


def do_quant(args):
    dataset = reader.Dataset(args)

    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    dataset = reader.Dataset(args)
    generator = dataset.file_reader(args.test_data, mode="test")
    
    start = time.time()
    ptq = PostTrainingQuantization(
            executor=exe,
            sample_generator=generator,
            model_dir=args.model_load_path,
            model_filename=None,
            params_filename=None,
            batch_size=args.batch_size,
            batch_nums=args.batch_num if args.batch_num > 0 else None,
            algo="KL",
            quantizable_op_type=["mul"],
            activation_quantize_type="range_abs_max",
            weight_quantize_type="channel_wise_abs_max",)
    quantized_program = ptq.quantize()
    ptq.save_quantized_model(args.model_save_path)
    times = time.time() - start
    print("It takes " + str(times) + "s. \n\n")

if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    check_cuda(args.use_cuda)
    check_version()
    do_quant(args)
