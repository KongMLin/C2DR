import pandas as pd
import torch
import copy
from torch import nn
import numpy as np
from huawei_2022.fuxictr.pytorch.models import BaseModel
from huawei_2022.fuxictr.pytorch.layers import EmbeddingLayer, EmbeddingDictLayer, MLP_Layer, Dice
import logging
import os, sys
from torch.nn.functional import binary_cross_entropy
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MADD(BaseModel):
    def __init__(self,
                feature_map_u,
                feature_map_b,
                feature_map_a,
                # b_hist = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003'],
                # a_hist = ['u_newsCatInterests',
                #     'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news'],
                #  b_hist=['movie_hist'],
                #  a_hist=['book_hist'],
                 b_hist=['hist'],
                 a_hist=['feeds_hist'],
                task='binary_classification',
                embedding_dim=128,
                dnn_hidden_units=[128],
                net_dropout=0,
                batch_norm=False,
                learning_rate=1e-3,
                tensorboard = None,
                **kwargs):
        super(MADD, self).__init__(feature_map_u,**kwargs)
        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        self.feature_map_u = feature_map_u
        self.feature_map_b = feature_map_b
        self.feature_map_a = feature_map_a

        self.user_feature_length = feature_map_u.input_length
        self.b_feature_length = feature_map_b.input_length
        self.a_feature_length = feature_map_a.input_length

        self.full_length = len(feature_map_u.feature_specs)
        self.user_length = self.full_length - len(b_hist) - len(a_hist)
        self.a_length = len(a_hist)
        self.b_length = len(b_hist)
        self.b_hist = b_hist
        self.a_hist = a_hist
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim)
        self.embedding_layer_a = EmbeddingDictLayer(feature_map_a, embedding_dim)
        self.embedding_layer_b = EmbeddingDictLayer(feature_map_b, embedding_dim)

        # 用于将user分为user_a user_b
        self.domain_user_split = nn.Linear(3, 6)

        self.size = [embedding_dim, embedding_dim, embedding_dim//2, embedding_dim//4]

        ###### private encoder A #####
        self.priA = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        ####### private encoder B #####
        self.priB = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        ####### shared encoder #######
        self.shared = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )

        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
            nn.ReLU(True),
            nn.Linear(in_features=self.size[3], out_features=2),
        )

        self.shared_decoder = nn.Sequential(
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0])
        )

        ####### domainA items autoencoder #######
        self.encoderA = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        self.decoderA = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0]),
        )
        ####### domainB items autoencoder #######
        self.encoderB = nn.Sequential(
            nn.Linear(in_features=self.size[0], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[3]),
        )
        self.decoderB = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[2]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[2], out_features=self.size[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.size[1], out_features=self.size[0]),
        )

        ####### domainA predict #######
        self.predict_user_A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.predict_item_A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        ####### domainB predict #######
        self.predict_user_B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.predict_item_B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]*2),
            nn.ReLU(),
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
        )

        self.A2B = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
        )
        self.B2A = nn.Sequential(
            nn.Linear(in_features=self.size[3], out_features=self.size[3]),
        )
        ####### domainA Attention #######
        self.attention1A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec1A =  nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias1A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention2A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec2A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias2A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention3A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec3A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias3A = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention4A = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec4A = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias4A = nn.Parameter(torch.FloatTensor([0]), True)

        ####### domainB Attention #######
        self.attention1B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec1B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias1B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention2B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec2B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias2B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention3B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec3B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias3B = nn.Parameter(torch.FloatTensor([0]), True)

        self.attention4B = nn.Sequential(
            nn.Linear(in_features=self.size[3]*2, out_features=self.size[3]),
            nn.ReLU()
        )
        self.atVec4B = nn.Parameter(torch.randn(self.size[3]), True)
        self.atBias4B = nn.Parameter(torch.FloatTensor([0]), True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = LossFn()
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs, p=0.5):
        if self.training:
            if self.p < 1:
                self.p += 1 / 15000
                self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        X, y = self.inputs_to_device(inputs)
        emb_itemA = self.embedding_layer_a(X[:, self.user_feature_length:self.user_feature_length + self.a_feature_length])
        emb_itemA = self.embedding_layer_a.dict2tensor(emb_itemA)
        emb_itemA = emb_itemA.sum(dim=1) / self.a_feature_length

        emb_itemB = self.embedding_layer_b(X[:, self.user_feature_length + self.a_feature_length:])
        emb_itemB = self.embedding_layer_b.dict2tensor(emb_itemB)
        emb_itemB = emb_itemB.sum(dim=1) / self.b_feature_length

        X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
        X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
        X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length

        for feature_name, feature in X_u.items():
            # print(feature_name,feature.shape)
            if feature_name in self.b_hist:
                tmp_feature = torch.mean(feature, dim=1)
                X_u[feature_name] = tmp_feature
            if feature_name in self.a_hist:
                tmp_feature = torch.mean(feature, dim=1)
                X_u[feature_name] = tmp_feature

        X_u = self.embedding_layer_u.dict2tensor(X_u)
        X_a_hist = X_u[:, self.user_length:self.user_length + self.a_length].sum(dim=1) / self.a_length
        X_b_hist = X_u[:, self.user_length + self.a_length:].sum(dim=1) / self.b_length
        X_u = torch.cat([X_user.unsqueeze(1), X_a_hist.unsqueeze(1), X_b_hist.unsqueeze(1)], dim=1)
        # X_u = torch.mean(X_u,dim=1)
        # print(X_u.shape)
        X_ab = self.domain_user_split(X_u.transpose(2, 1)).transpose(2, 1)
        X_a = torch.mean(X_ab[:, :3], dim=1)
        X_b = torch.mean(X_ab[:, 3:], dim=1)

        X_shared_a = self.shared(X_a)
        X_shared_b = self.shared(X_b)

        reversed_shared_code_a = ReverseLayerF.apply(X_shared_a, p)
        reversed_shared_code_b = ReverseLayerF.apply(X_shared_b, p)

        pred_domain_a = self.shared_encoder_pred_domain(reversed_shared_code_a)
        pred_domain_b = self.shared_encoder_pred_domain(reversed_shared_code_b)

        pria_uembedding = self.priA(X_a)
        # print(pria_uembedding.shape)
        # print(emb_itemA.shape)

        encode_iembedding_a = self.encoderA(emb_itemA)

        decode_iembedding_a = self.decoderA(encode_iembedding_a)
        sharedecoder_a = self.shared_decoder(torch.cat((pria_uembedding, X_shared_a), dim=1))
        pribtoa = self.B2A(self.priB(X_b))

        tmp_p = torch.cat((pria_uembedding, pribtoa), dim=1)
        Weight1 = torch.exp(torch.sum(self.attention1A(tmp_p).mul(self.atVec1A), dim=1) + self.atBias1A).unsqueeze(dim=1)
        Weight2 = torch.exp(torch.sum(self.attention2A(tmp_p).mul(self.atVec2A), dim=1) + self.atBias2A).unsqueeze(dim=1)
        pri_uembedding = Weight1*pria_uembedding + Weight2*pribtoa

        tmp_ps = torch.cat((pri_uembedding, X_shared_a), dim=1)
        Weight3 = torch.exp(torch.sum(self.attention3A(tmp_ps).mul(self.atVec3A), dim=1) + self.atBias3A).unsqueeze(dim=1)
        Weight4 = torch.exp(torch.sum(self.attention4A(tmp_ps).mul(self.atVec4A), dim=1) + self.atBias4A).unsqueeze(dim=1)
        domainu_embedding = Weight3*pri_uembedding + Weight4*X_shared_a
        # print(domainu_embedding.shape,decode_iembedding_a.shape)
        predicta = torch.sum(self.predict_user_A(domainu_embedding).mul(self.predict_item_A(encode_iembedding_a)), dim=1)

        prib_uembedding = self.priB(X_b)
        encode_iembedding_b = self.encoderB(emb_itemB)
        decode_iembedding_b = self.decoderB(encode_iembedding_b)
        sharedecoder_b = self.shared_decoder(torch.cat((prib_uembedding, X_shared_b), dim=1))
        # print(X_shared_b.shape)
        priatob = self.A2B(self.priA(X_a))

        tmp_p = torch.cat((prib_uembedding, priatob), dim=1)
        Weight1 = torch.exp(torch.sum(self.attention1B(tmp_p).mul(self.atVec1B), dim=1) + self.atBias1B).unsqueeze(dim=1)
        Weight2 = torch.exp(torch.sum(self.attention2B(tmp_p).mul(self.atVec2B), dim=1) + self.atBias2B).unsqueeze(dim=1)
        pri_uembedding = Weight1*prib_uembedding + Weight2*priatob

        tmp_ps = torch.cat((pri_uembedding, X_shared_b), dim=1)
        Weight3 = torch.exp(torch.sum(self.attention3B(tmp_ps).mul(self.atVec3B), dim=1) + self.atBias3B).unsqueeze(dim=1)
        Weight4 = torch.exp(torch.sum(self.attention4B(tmp_ps).mul(self.atVec4B), dim=1) + self.atBias4B).unsqueeze(dim=1)
        domainu_embedding = Weight3*pri_uembedding + Weight4*X_shared_b

        predictb = torch.sum(self.predict_user_B(domainu_embedding).mul(self.predict_item_B(encode_iembedding_b)), dim=1)
        return {'pred_domain_a': pred_domain_a, 'pred_domain_b': pred_domain_b, 'shared_a': X_shared_a, 'shared_b': X_shared_b, \
                    'pria_uembedding': pria_uembedding, 'prib_uembedding': prib_uembedding, 'sharedecoder_a': sharedecoder_a, \
                        'sharedecoder_b': sharedecoder_b, 'X_a': X_a, 'X_b': X_b, 'itemA_emb': emb_itemA, 'itemB_emb': emb_itemB, \
                            'decode_iembedding_a': decode_iembedding_a, 'decode_iembedding_b': decode_iembedding_b, \
                'pred_a': predicta,  'pred_b': predictb, 'ya_true': y[:,0], 'yb_true': y[:,1]}

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 2).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():

            pred_a = []
            pred_b = []
            ya_true = []
            yb_true = []

            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)

                pred_a.extend(return_dict["pred_a"].data.cpu().numpy().reshape(-1))
                pred_b.extend(return_dict["pred_b"].data.cpu().numpy().reshape(-1))
                ya_true.extend(return_dict["ya_true"].data.cpu().numpy().reshape(-1))
                yb_true.extend(return_dict["yb_true"].data.cpu().numpy().reshape(-1))

            pred_a = np.array(pred_a, np.float64)
            pred_b = np.array(pred_b, np.float64)
            # print(pred_a.shape,pred_b.shape)

            ya_true = np.array(ya_true, np.float64)
            yb_true = np.array(yb_true, np.float64)
            print(ya_true)
            ya_true[0] = 1
            yb_true[0] = 1
            # if self.tensorboard is not None:
            #     self.tensorboard = self.tensorboard.append(
            #         pd.DataFrame({'v_u': v_u, 'v_s': v_s, 'v_t': v_t,'y_true':y_true}), ignore_index=True)
            # self.tensorboard['v_u'] = v_u1
            # self.tensorboard['v_s'] = v_s1
            # self.tensorboard['v_t'] = v_t1
            # print(ya_true.shape,yb_true.shape,pred_a.shape,pred_b.shape)
            val_logs = self.evaluate_metrics(ya_true, pred_a, self._validation_metrics)
            print((yb_true == 1).sum())
            print((yb_true == 0).sum())
            val_logs2 = self.evaluate_metrics(yb_true, pred_b, self._validation_metrics)
            # print(y_pred_source.shape,y_true_source.shape,y_pred_target.shape,y_true_target.shape)
            # auc_source = self.evaluate_metrics(y_true_source,y_pred_source, self._validation_metrics)['AUC']
            # auc_target = self.evaluate_metrics(y_true_target,y_pred_target, self._validation_metrics)['AUC']
            # A_distance = 2 * abs(auc_source - auc_target)
            # print(A_distance)
            return val_logs, val_logs2, pred_a, ya_true, pred_b, yb_true

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse

class LossFn(nn.Module):
    def __init__(self) -> None:
        super(LossFn, self).__init__()
        self.loss_classification = nn.MSELoss()
        self.loss_recon = SIMSE()
        self.loss_diff = DiffLoss()
        self.loss_autoencoder = nn.MSELoss()
        self.loss_similarity = torch.nn.CrossEntropyLoss()
    
    def forward(self, input, reduction= 'mean'):
        # loss_sim
        pred_domain_a = input['pred_domain_a']
        pred_domain_b = input['pred_domain_b']
        
        # loss_diff
        shared_a = input['shared_a']
        shared_b = input['shared_b']
        pria = input['pria_uembedding']
        prib = input['prib_uembedding']

        # loss_recon
        sharedecoder_a = input['sharedecoder_a']
        sharedecoder_b = input['sharedecoder_b']
        X_a = input['X_a']
        X_b = input['X_b']

        # loss_encoder
        itemA_emb = input['itemA_emb']
        itemB_emb = input['itemB_emb']
        decode_iembedding_a = input['decode_iembedding_a']
        decode_iembedding_b = input['decode_iembedding_b']

        # loss_class
        predicta = input['pred_a']
        predictb = input['pred_b']
        label_a = input['ya_true']
        label_b = input['yb_true']

        loss_sim = self.loss_similarity(pred_domain_a, torch.zeros_like(pred_domain_a))
        loss_sim += self.loss_similarity(pred_domain_b, torch.ones_like(pred_domain_a))

        loss_diff = self.loss_diff(shared_a, pria)
        loss_diff += self.loss_diff(shared_b, prib)

        loss_recon = self.loss_recon(sharedecoder_a, X_a)
        loss_recon += self.loss_recon(sharedecoder_b, X_b)
        # print(itemA_emb.shape,decode_iembedding_a.shape)
        loss_encoder = self.loss_autoencoder(itemA_emb, decode_iembedding_a)
        loss_encoder += self.loss_autoencoder(itemB_emb, decode_iembedding_b)

        loss_class = self.loss_classification(predicta.squeeze(), label_a)
        loss_class += self.loss_classification(predictb.squeeze(), label_b)

        return loss_sim + loss_diff + loss_recon + loss_encoder + loss_class

