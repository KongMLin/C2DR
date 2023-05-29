# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import torch
torch.set_num_threads(20) # 限制torch使用的进程数
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap, FeatureEncoder
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import os
from pathlib import Path
# 过滤警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# 直接在主实验函数中实现，将训练集划分为训练和验证，提交的数据集划为测试，测试所用标签随机即可（测试结果无参考意义）
# 这里一定要保证训练和测试的特征与标签要一样，因为解析的方法是一样的

if __name__ == '__main__':
    experments = [
        # "widedeep_huawei_process",
        "DIN_base",
    ]

    for experment_id in experments:
        parser = argparse.ArgumentParser()
        # 指定模型依赖框架、运行设备
        parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
        parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
        # 指定config目录文件夹所在
        parser.add_argument('--config', type=str, default='./config/hauwei',
                            help='The config directory.')
        # 指定model_config.yaml中的实验id（以此来控制实验）
        parser.add_argument('--expid', type=str, default=experment_id, help='The experiment id to run.')

        # 将上述相关参数写入json文件，并保存
        args = vars(parser.parse_args())
        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)
        params['gpu'] = args['gpu']
        params['version'] = args['version']
        set_logger(params)
        logging.info(print_to_json(params))
        seed_everything(seed=params['seed'])

        # preporcess the dataset（预处理数据集）
        dataset = params['dataset_id'].split('_')[0].lower()
        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        if params.get("data_format") == 'h5': # load data from h5（从h5文件中加载数据集）
            feature_map = FeatureMap(params['dataset_id'], data_dir, params['version'])
            json_file = os.path.join(os.path.join(params['data_root'], params['dataset_id']), "feature_map.json")
            if os.path.exists(json_file):
                feature_map.load(json_file)
            else:
                raise RuntimeError('feature_map not exist!')
        else: # load data from csv（如果首次读入的是CSV文件，则进行h5文件的构建）
            try:
                feature_encoder = getattr(datasets, dataset).FeatureEncoder(**params)
            except:
                feature_encoder = FeatureEncoder(**params)
            if os.path.exists(feature_encoder.json_file):
                feature_encoder.feature_map.load(feature_encoder.json_file)
            else: # Build feature_map and transform h5 data（构建特征映射并转换为h5数据集）
                datasets.build_dataset(feature_encoder, **params)
            params["train_data"] = os.path.join(data_dir, 'train*.h5')
            params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
            params["test_data"] = os.path.join(data_dir, 'test*.h5')
            feature_map = feature_encoder.feature_map

        # get train and validation data（获取训练与验证数据集，均来源于训练集）
        train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **params)
        print(train_gen)
        # initialize model（初始化模型，并将相关参数设置归于模型中）
        model_class = getattr(models, params['model'])
        model = model_class(feature_map, **params)
        # print number of parameters used in model（打印模型总训练参数量）
        model.count_parameters()
        # fit the model（拟合模型）
        model.fit_generator(train_gen, validation_data=valid_gen, **params)

        # load the best model checkpoint（载入最佳模型参数）
        logging.info("Load best model: {}".format(model.checkpoint))
        model.load_weights(model.checkpoint)

        # get evaluation results on validation（获取模型在验证集上的结果）
        logging.info('****** Validation evaluation ******')
        valid_result, y_pred, y_true = model.evaluate_generator(valid_gen)
        del train_gen, valid_gen
        # gc.collect()

        # get evaluation results on test（获取模型在测试集上的结果）
        logging.info('******** Test evaluation ********')
        test_gen = datasets.h5_generator(feature_map, stage='test', **params)
        if test_gen:
            test_result, y_pred, y_true = model.evaluate_generator(test_gen)
            # print(y_pred.tolist(), y_true.tolist())
        else:
            test_gen = {}
        # save the results to csv（将当前批次运行的结果保存至csv文件中）
        result_file = Path(args['config']).name.replace(".yaml", "") + '.csv'
        with open(result_file, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                        ' '.join(sys.argv), experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))



    import pandas as pd

    test_data = pd.read_csv(r"C:\Users\Administrator\PycharmProjects\pythonProject2\huawei\data\train\processed_test_data_ads_withfakelabel.csv", dtype=str)
    result = {"log_id":test_data["log_id"].tolist(), "pctr":[round(i, 6) for i in y_pred]}
    result = pd.DataFrame(result)
    result.to_csv(r"C:\Users\Administrator\PycharmProjects\pythonProject2\huawei\data\submission.csv")

    test_gen = datasets.h5_generator(feature_map, stage='test', **params)