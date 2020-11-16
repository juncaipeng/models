# 准备

具体参考[链接](https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/lexical_analysis)进行准备工作。

安装Paddle和PaddleHub。

Clone paddlemodels repo，切换到release1.8。

下载数据集文件，解压后会生成 ./data/ 文件夹。

`python downloads.py dataset`

下载预训练lac模型。

`python downloads.py lac`

测试预训练lac模型精度。

`bash run.sh eval`

# 导出模型

导出预测模型，该模型不含精度计算op。

```
export PYTHONIOENCODING=UTF-8
python inference_model.py \
        --init_checkpoint ./model_baseline \
        --inference_save_dir ./inference_model
```

如果需要导出包含精度计算op的模型，需要修改`inference_model.py`中的模型输入和输出。


# 量化

产出包含精度计算op的GRU量化模型

`python quant_gru_acc.py`

产出不包含精度计算op的GRU量化模型，需要将返回lodtensor的reader直接传入到量化api中，修改源码直接使用reader数据进行前向计算，然后再执行：

`python quant_gru_noacc.py`
