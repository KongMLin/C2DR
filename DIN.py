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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Function
import seaborn as sns
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DADINv2(BaseModel):
    def __init__(self,
                 feature_map_u,
                 feature_map_t,
                 feature_map_s,
                 target_hist = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003'],
                 source_hist = ['u_newsCatInterests',
                    'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news'],
                 task='binary_classification',
                 embedding_dim=128,
                 dnn_hidden_units=[128],
                 net_dropout=0,
                 batch_norm=False,
                 learning_rate=1e-3,
                 tensorboard = None,
                 **kwargs):
        super(DADINv2, self).__init__(feature_map_u,**kwargs)

        self.seq_agg = kwargs['seq_agg']
        self.interset_attn = kwargs['interest_attn']
        self.bi_pooling = kwargs['bi_pooling']
        self.dnn = kwargs['dnn']
        self.domain_ag = kwargs['domain_agnostic']
        self.domain_pred = kwargs['domain_pred']
        self.c_domain_pred = kwargs['c_domain_pred']
        self.tensorboard = tensorboard
        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1
        #amazon dataset
        # target_hist = ['hist']
        # source_hist = ['feeds_hist']
        self.feature_map_u = feature_map_u
        self.feature_map_t = feature_map_t
        self.feature_map_s = feature_map_s

        self.target_hist = target_hist
        self.source_hist = source_hist

        self.user_feature_length = feature_map_u.input_length
        self.target_feature_length = feature_map_t.input_length
        self.source_feature_length = feature_map_s.input_length

        # print(feature_map_u.num_features,feature_map_u.feature_specs)
        # print(feature_map_u.input_length)
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim)
        self.embedding_layer_s = EmbeddingDictLayer(feature_map_s, embedding_dim)
        self.embedding_layer_t = EmbeddingDictLayer(feature_map_t, embedding_dim)

        self.attention_layers_item_s = ItemAttentionLayer(embedding_dim=embedding_dim,device=kwargs['gpu'])
        self.attention_layers_item_t = ItemAttentionLayer(embedding_dim=embedding_dim,device=kwargs['gpu'])

        #定义变量，便于处理每个field内的特征，pooling操作
        self.full_length = len(feature_map_u.feature_specs)
        self.user_length = self.full_length - len(target_hist) - len(source_hist)
        self.source_length = len(source_hist)
        self.target_length = len(target_hist)
        self.attention_layers_interest = InterestAttentionLayer(embedding_dim=embedding_dim)
        self.bi_interaction_pooling = BiInteractionPooling()

        self.hidden_layers = [embedding_dim] + dnn_hidden_units + [embedding_dim]
        self.dnn_layers = nn.ModuleList([
            nn.Linear(in_features=layer[0], out_features=layer[1])\
                for layer in list(zip(self.hidden_layers[:-1], self.hidden_layers[1:]))
        ])

        self.domain_agnostic = DomainAgnosticNet(embedding_dim=embedding_dim)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(embedding_dim, 100))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 1))
        self.class_classifier.add_module('c_softmax', nn.Sigmoid())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(embedding_dim, 1))
        self.domain_classifier.add_module('d_softmax', nn.Sigmoid())

        self.domain_classifier_c0 = copy.deepcopy(self.domain_classifier)
        self.domain_classifier_c1 = copy.deepcopy(self.domain_classifier)
        #lambda2= 0.5, 1
        #lambda3= 0.5, 1
        loss_fn = Loss_fn(domain_pred=self.domain_pred,c_domain_pred=self.c_domain_pred,lambda2=0.5,lambda3=0.5)
        self.compile(kwargs["optimizer"], loss=loss_fn, lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
        print(learning_rate)
    def forward(self, inputs):

        if self.training:
            if self.p < 1:
                self.p += 1 / 15000
                self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        X, y = self.inputs_to_device(inputs)

        source_mask = [y[:, -1] == 0]
        target_mask = [y[:, -1] == 1]

        X_s, y_s = X[source_mask], y[source_mask]
        X_t, y_t = X[target_mask], y[target_mask]
        # print(X_t[:, self.user_feature_length + self.source_feature_length:])
        X_all = []
        y_all = []
        if y_s.shape[0] != 0:
            X_s_u = self.embedding_layer_u(X_s[:, :self.user_feature_length])
            X_s_s = self.embedding_layer_s(X_s[:, self.user_feature_length:self.user_feature_length + self.source_feature_length])
            X_s_s = self.embedding_layer_s.dict2tensor(X_s_s)
            # X_s_s = self.source_linear(X_s_s.flatten(start_dim=1))
            # print(X_s_u)
            X_s_s = torch.sum(X_s_s,dim=1)/len(self.feature_map_s.feature_specs)
            X_s_u_emb = self.embedding_layer_u.dict2tensor(X_s_u,user_feature=self.user_length)
            X_user = X_s_u_emb[:,:self.user_length].sum(dim=1)/self.user_length
            for feature_name, feature in X_s_u.items():
                # print(feature_name,feature.shape)
                if feature_name in self.source_hist:
                    if not self.seq_agg:
                        tmp_feature = torch.mean(feature, dim=1)
                    else:
                        tmp_feature = self.attention_layers_item_s(feature, X_s_s, X_user)
                    X_s_u[feature_name] = tmp_feature

                if feature_name in self.target_hist:
                    if not self.seq_agg:
                        tmp_feature = torch.mean(feature, dim=1)
                    else:
                        tmp_feature = self.attention_layers_item_t(feature, X_s_s, X_user)
                    X_s_u[feature_name] = tmp_feature
                # print(feature_name,X_s_u[feature_name].shape)
            X_s_u = self.embedding_layer_u.dict2tensor(X_s_u)
            X_s_hist = X_s_u[:,self.user_length:self.user_length+self.source_length].sum(dim=1)/self.source_length
            X_t_hist = X_s_u[:,self.user_length+self.source_length:].sum(dim=1)/self.target_length
            X_s = torch.cat([X_user.unsqueeze(1),X_s_hist.unsqueeze(1),X_t_hist.unsqueeze(1), X_s_s.unsqueeze(1)], dim=1)
            if not self.interset_attn:
                X_s = X_s.reshape(X_s.shape[0], -1)
            else:
                X_s,v_u,v_s,v_t = self.attention_layers_interest(X_s)
            X_all.append(X_s)
            y_all.append(y_s)

        if y_t.shape[0] != 0:

            X_t_u = self.embedding_layer_u(X_t[:, :self.user_feature_length])
            X_t_t = self.embedding_layer_t(X_t[:, self.user_feature_length + self.source_feature_length:])
            X_t_t = self.embedding_layer_t.dict2tensor(X_t_t)
            # X_t_t = self.target_linear(X_t_t.flatten(start_dim=1))
            X_t_t = torch.sum(X_t_t, dim=1) / len(self.feature_map_t.feature_specs)
            X_t_u_emb = self.embedding_layer_u.dict2tensor(X_t_u,user_feature=self.user_length)
            X_user = X_t_u_emb[:,:self.user_length].sum(dim=1)/self.user_length
            for feature_name, feature in X_t_u.items():
                if feature_name in self.source_hist:
                    if self.seq_agg:
                        tmp_feature = self.attention_layers_item_s(feature, X_t_t, X_user)
                    else:
                        tmp_feature = torch.mean(feature, dim=1)
                    X_t_u[feature_name] = tmp_feature
                if feature_name in self.target_hist:
                    if self.seq_agg:
                        tmp_feature = self.attention_layers_item_t(feature, X_t_t, X_user)
                    else:
                        tmp_feature = torch.mean(feature, dim=1)
                    X_t_u[feature_name] = tmp_feature
            X_t_u = self.embedding_layer_u.dict2tensor(X_t_u)
            X_s_hist = X_t_u[:,self.user_length:self.user_length+self.source_length].sum(dim=1)/self.source_length
            X_t_hist = X_t_u[:,self.user_length+self.source_length:].sum(dim=1)/self.target_length
            X_t= torch.cat([X_user.unsqueeze(1),X_s_hist.unsqueeze(1),X_t_hist.unsqueeze(1), X_t_t.unsqueeze(1)], dim=1)
            if self.interset_attn:
                X_t,v_u,v_s,v_t = self.attention_layers_interest(X_t)
            else:
                X_t = X_t.reshape(X_t.shape[0], -1)
            X_all.append(X_t)
            y_all.append(y_t)
        X = torch.cat(X_all, dim=0)
        y = torch.cat(y_all,dim=0)

        if self.bi_pooling:
            X = self.bi_interaction_pooling(X)
        else:
            X = X.reshape(X.shape[0], 4, -1)
            X = torch.mean(X, dim=1)
        X = X.squeeze(1)

        if self.dnn:
            for dnn in self.dnn_layers:
                X = dnn(X)
                X = torch.relu(X)
            X = torch.dropout(X,p=0.2,train=self.training)

        if self.domain_ag:
            X_DA = self.domain_agnostic(X)  # 混淆之后只存在领域无关信息
            X = X + X_DA
            # X = X_DA
            reverse_X = ReverseLayerF.apply(X_DA, self.alpha)
        else:
            reverse_X = ReverseLayerF.apply(X, self.alpha)

        pred = self.class_classifier(X)
        pred_d = self.domain_classifier(reverse_X)
        pred_d_c = []
        
        pred_d_c.append(self.domain_classifier_c0(reverse_X*(1-pred)))
        pred_d_c.append(self.domain_classifier_c1(reverse_X*pred))

        pred_dict = {'y_pred':pred,'domain_output':pred_d,'y_true':y[:,0].unsqueeze(-1),'domain_true':y[:,1].unsqueeze(-1), 'domain_c_output':pred_d_c,
                     'v_u':v_u,'v_s':v_s,'v_t':v_t,'x_DA':X_DA,'x_spec':X}

        if not self.training:
            print(pred_dict['y_pred'][:10])
            print(pred_dict['y_true'][:10])

        return pred_dict

    def evaluate_generator(self, data_generator):
        self.eval() # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            y_pred_target = []
            y_true_target = []
            y_pred_source = []
            y_true_source = []
            v_u = []
            v_s = []
            v_t = []
            x_DA = []
            x_spec = []
            domain = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(return_dict["y_true"].data.cpu().numpy().reshape(-1))
                domain.extend(return_dict["domain_true"].data.cpu().numpy().reshape(-1))
                x_DA.extend(return_dict["x_DA"].data.cpu().numpy().reshape(-1,self.embedding_dim))
                x_spec.extend(return_dict["x_spec"].data.cpu().numpy().reshape(-1,self.embedding_dim))
                # print('domain:',domain)
                y_pred_target.extend(return_dict['y_pred'].data.cpu().numpy().reshape(-1)[domain == 1])
                y_true_target.extend(return_dict['y_true'].data.cpu().numpy().reshape(-1)[domain == 1])
                y_pred_source.extend(return_dict['y_pred'].data.cpu().numpy().reshape(-1)[domain == 0])
                y_true_source.extend(return_dict['y_true'].data.cpu().numpy().reshape(-1)[domain == 0])
                v_u.extend(return_dict['v_u'].data.cpu().numpy())
                v_s.extend((return_dict['v_s']).data.cpu().numpy())
                v_t.extend((return_dict['v_t']).data.cpu().numpy())
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)

            x_DA = np.array(x_DA,np.float64)
            x_spec = np.array(x_spec,np.float64)
            domain = np.array(domain, np.float64)
            # np.save(r'D:\test\amazon_x_DA.npy',x_DA)
            # np.save(r'D:\test\amazon_x_spec.npy', x_spec)
            # np.save(r'D:\test\amazon_yd.npy', domain)

            y_pred_target = np.array(y_pred_target, np.float64)
            y_true_target = np.array(y_true_target, np.float64)
            y_pred_source = np.array(y_pred_source, np.float64)
            y_true_source = np.array(y_true_source, np.float64)
            v_u = np.array(v_u,np.float64).reshape(-1)
            v_s = np.array(v_s,np.float64).reshape(-1)
            v_t = np.array(v_t,np.float64).reshape(-1)
            # v_u1 = [x.item() for x in v_u]
            # v_s1 = [x.item() for x in v_s]
            # v_t1 = [x.item() for x in v_t]
            print(v_u.shape,v_t.shape,v_s.shape)
            # if self.tensorboard is not None:
            #     self.tensorboard = self.tensorboard.append(
            #         pd.DataFrame({'v_u': v_u, 'v_s': v_s, 'v_t': v_t,'y_true':y_true}), ignore_index=True)
                # self.tensorboard['v_u'] = v_u1
                # self.tensorboard['v_s'] = v_s1
                # self.tensorboard['v_t'] = v_t1
            val_logs = self.evaluate_metrics(y_true, y_pred, self._validation_metrics)
            # print(y_pred_source.shape,y_true_source.shape,y_pred_target.shape,y_true_target.shape)
            # auc_source = self.evaluate_metrics(y_true_source,y_pred_source, self._validation_metrics)['AUC']
            # auc_target = self.evaluate_metrics(y_true_target,y_pred_target, self._validation_metrics)['AUC']
            # A_distance = 2 * abs(auc_source - auc_target)
            # print(A_distance)
            return val_logs, y_pred, y_true
    #
    def add_loss(self, inputs, reduction="mean"):
        return_dict = self.forward(inputs)

        loss, loss_list = self.loss_fn(return_dict, reduction=reduction)
        return loss, loss_list

    def get_total_loss(self, inputs):
        total_loss, loss_list = self.add_loss(inputs)

        total_loss = total_loss + self.add_regularization()
        return total_loss, loss_list


    def train_one_epoch(self, data_generator, epoch):
        epoch_loss = 0

        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            loss, loss_list = self.get_total_loss(batch_data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(batch_index)
            if self._stop_training:
                break
            if self.tensorboard is not None:
            #     self.tensorboard.add_scalar('Loss/loss', loss.item(), batch_index + epoch * len(data_generator))
            #     self.tensorboard.add_scalar('Loss/loss1', loss_list[0].item(),
            #                                 batch_index + epoch * len(data_generator))
            #     self.tensorboard.add_scalar('Loss/loss2', loss_list[1].item(),
            #                                 batch_index + epoch * len(data_generator))
            #     self.tensorboard.add_scalar('Loss/loss3', loss_list[2].item(),
            #                                 batch_index + epoch * len(data_generator))
                self.tensorboard = self.tensorboard.append(
                    {'loss': loss.item(), 'loss1': loss_list[0].item(), 'loss2': loss_list[1].item(),
                     'loss3': loss_list[2].item()}, ignore_index=True)
        return epoch_loss / self._batches_per_epoch

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 2).to(self.device)
        self.batch_size = y.size(0)
        return X, y

class ItemAttentionLayer(nn.Module):
    def __init__(self,
                 h=4,
                 device=None,
                embedding_dim=64):
        super(ItemAttentionLayer, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.h = embedding_dim * h
        self.M = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_s = nn.Linear(4 * embedding_dim, h)
        self.relu = nn.ReLU()
        self.h_s = nn.Linear(h, 1)
        self.softmax = nn.Softmax()

    def forward(self, hist, X_i, X_user):
        bs, n, _ = hist.size()

        a = torch.zeros([bs, n]).to(self.device)
        for i in range(n):
            r_si = hist[:, i]
            tmp = self.M(r_si * X_i)
            tmp = torch.cat([r_si, X_i, X_user, tmp], dim=1)
            a_i = self.h_s(self.relu(self.W_s(tmp)))
            a[:, i] = a_i.squeeze(-1)

        a = self.softmax(a)

        return torch.sum(hist * a.unsqueeze(-1), dim=1)

class InterestAttentionLayer(nn.Module):
    def __init__(self,embedding_dim=64):
        super(InterestAttentionLayer, self).__init__()
        self.relu = nn.ReLU()
        self.Linear_V_u = nn.Linear(4 * embedding_dim, embedding_dim, bias=False)
        self.Linear_G_u = nn.Linear(embedding_dim, 1)
        self.Linear_V_s = nn.Linear(4 * embedding_dim, embedding_dim, bias=False)
        self.Linear_G_s = nn.Linear(embedding_dim, 1)
        self.Linear_V_t = nn.Linear(4 * embedding_dim, embedding_dim, bias=False)
        self.Linear_G_t = nn.Linear(embedding_dim, 1)

    def forward(self, X):

        tmp = X.flatten(start_dim=1)
        v_u = torch.exp(self.Linear_G_u(self.relu(self.Linear_V_u(tmp))))
        v_s = torch.exp(self.Linear_G_s(self.relu(self.Linear_V_s(tmp))))
        v_t = torch.exp(self.Linear_G_s(self.relu(self.Linear_V_t(tmp))))

        X_user = v_u * X[:,0]
        X_s_hist = v_s * X[:, 1]
        X_t_hist = v_t * X[:, 2]
        X_i = X[:, -1]

        return torch.cat([X_user, X_s_hist, X_t_hist, X_i], dim=1),v_u,v_s,v_t


class DomainAgnosticNet(nn.Module):
    def __init__(self,embedding_dim=64):
        super(DomainAgnosticNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim,embedding_dim)
        self.activiation = nn.Sigmoid()

    def forward(self, input):
        tmp = self.linear(input.flatten(start_dim=1))
        tmp = self.activiation(tmp)
        return tmp

class BiInteractionPooling(nn.Module):
    def __init__(self) -> None:
        super(BiInteractionPooling, self).__init__()
    
    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs, 4, -1)
        # x (batch_size, 4, embedding_size) -> cross_term (batch_size, 1, embedding_size)
        concated_embeds_value = x
        square_of_sum = torch.square(torch.sum(concated_embeds_value, dim=1, keepdim=True))
        sum_of_square = torch.sum(torch.square(concated_embeds_value), dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

class Loss_fn(nn.Module):
    def __init__(self, classes=2, alpha=0.5,
                 domain_pred=True,
                 c_domain_pred=True,lambda2=1,lambda3=1):
        super(Loss_fn,self).__init__()
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.classes = classes
        self.alpha = [alpha, 1-alpha]
        self.domain_pred = domain_pred
        self.c_domain_pred = c_domain_pred
        # self.fn1 = F.binary_cross_entropy()
        # self.fn2 = F.binary_cross_entropy()
    def forward(self,input,reduction=None):

        y_pred = input['y_pred']
        y_true = input['y_true']

        domain = input['domain_output']
        domain_c = input['domain_c_output']
        domain_true = input['domain_true']
        total_loss = 0

        loss_list = []

        # 分类损失
        loss1 = F.binary_cross_entropy(y_pred,y_true)
        total_loss += loss1
        loss_list.append(loss1)
        print('loss1:',loss1)
        if self.domain_pred:

            # 领域损失
            loss2 = F.binary_cross_entropy(domain,domain_true)
            print('loss2:',loss2)# 全局
            total_loss += loss2*self.lambda2
            loss_list.append(loss2)
        if self.c_domain_pred:

            loss3 = 0
            threshold = 0.05
            for i in range(self.classes):
                if i == 0:
                    mask = y_pred < threshold
                else:
                    mask = y_pred >= threshold
                if sum(mask) == 0:
                    continue
                loss3 += self.alpha[i] * binary_cross_entropy(domain_c[i][mask],domain_true[mask])
            print('loss3:', loss3)
            total_loss += loss3*self.lambda3
            loss_list.append(loss3)
        # loss3 = 0
        # for i in range(self.classes):
        #     loss3 += binary_cross_entropy(domain_c[i],domain_true)

        return total_loss, loss_list
