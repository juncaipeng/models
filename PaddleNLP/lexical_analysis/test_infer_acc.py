# -*- coding: UTF-8 -*-

import argparse
import sys
import os

import numpy as np
import paddle
import paddle.fluid as fluid

import creator
import reader
import utils
sys.path.append('../shared_modules/models/')
from model_check import check_cuda
from model_check import check_version

paddle.enable_static()

def test_inference_model(model_dir, text_list, dataset):
    """
    :param model_dir: model's dir
    :param text_list: a list of input text, which decode as unicode
    :param dataset:
    :return:
    """
    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # transfer text data to input tensor
    lod = []
    for text in text_list:
        lod.append(np.array(dataset.word_to_ids(text.strip())).astype(np.int64))
    base_shape = [[len(c) for c in lod]]
    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

    # for empty input, output the same empty
    if (sum(base_shape[0]) == 0):
        crf_decode = [tensor_words]
    else:
        # load inference model
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(
                 model_dir,
                 exe,
                 model_filename=args.model_filename,
                 params_filename=args.params_filename, )
            assert feed_target_names[0] == "words"
            print("Load inference model from %s" % (model_dir))

            # get lac result
            crf_decode = exe.run(
                inferencer,
                feed={feed_target_names[0]: tensor_words},
                fetch_list=fetch_targets,
                return_numpy=False,
                use_program_cache=True, )

    # parse the crf_decode result
    result = utils.parse_result(tensor_words, crf_decode[0], dataset)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, 'conf/args.yaml')
    model_g = utils.ArgumentGroup(parser, "model", "model configuration")
    model_g.add_arg("load_model_path", str, "", "")
    model_g.add_arg("model_filename", str, "", "")
    model_g.add_arg("params_filename", str, "", "")
    args = parser.parse_args()
    args.model_filename = None if \
            args.model_filename == "" else args.model_filename
    args.params_filename = None if \
            args.params_filename == "" else args.params_filename

    check_cuda(args.use_cuda)
    check_version()

    print("test inference model")
    dataset = reader.Dataset(args)
    test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']
    #test_data = [u'百度是一家高科技公司']
    test_inference_model(args.load_model_path, test_data, dataset)
