import torch
from torch import nn
from huawei_2022.fuxictr.pytorch.models import BaseModel
from huawei_2022.fuxictr.pytorch.layers import EmbeddingLayer, EmbeddingDictLayer, MLP_Layer, Dice
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy

class CoNet(BaseModel):
    def __init__(self,
                 feature_map_u,
                 feature_map_t,
                 feature_map_s,
                 task='binary_classification',
                 embedding_dim=128,
                 layer_num=3,
                 learning_rate=1e-3,
                 **kwargs):
        super(CoNet, self).__init__(feature_map_u,**kwargs)

        self.feature_map_u = feature_map_u
        self.feature_map_t = feature_map_t
        self.feature_map_s = feature_map_s

        self.user_feature_length = feature_map_u.input_length
        self.target_feature_length = feature_map_t.input_length
        self.source_feature_length = feature_map_s.input_length

        # print(feature_map_u.num_features,feature_map_u.feature_specs)
        # print(feature_map_u.input_length)
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim,sequence_pooling=True)
        self.embedding_layer_s = EmbeddingDictLayer(feature_map_s, embedding_dim)
        self.embedding_layer_t = EmbeddingDictLayer(feature_map_t, embedding_dim)

        self.layers = [CrossConnectionLayer(embedding_dim=embedding_dim,device=kwargs['gpu']) for _ in range(layer_num)]
        self.layers = nn.Sequential(*self.layers)

        self.target_classifier = nn.Sequential()
        self.target_classifier.add_module('c_fc1', nn.Linear(2*embedding_dim, 1))
        self.target_classifier.add_module('c_softmax', nn.Sigmoid())

        self.source_classifier = nn.Sequential()
        self.source_classifier.add_module('d_fc1', nn.Linear(2*embedding_dim, 1))
        self.source_classifier.add_module('d_softmax', nn.Sigmoid())
        self.loss_fn = Loss_fn()
        self.compile(kwargs["optimizer"], loss=self.loss_fn, lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)

        X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
        X_u = self.embedding_layer_u.dict2tensor(X_u)
        X_u = torch.sum(X_u, dim=1) / len(self.feature_map_u.feature_specs) # pooling

        X_s = self.embedding_layer_s(X[:, self.user_feature_length:self.user_feature_length+self.source_feature_length])
        X_s = self.embedding_layer_s.dict2tensor(X_s)
        X_s = torch.sum(X_s, dim=1)/len(self.feature_map_s.feature_specs)

        X_t = self.embedding_layer_t(X[:, self.user_feature_length + self.source_feature_length:])
        X_t = self.embedding_layer_t.dict2tensor(X_t)
        X_t = torch.sum(X_t, dim=1) / len(self.feature_map_t.feature_specs)
        
        X_s = torch.cat([X_u, X_s], dim=1)
        X_t = torch.cat([X_u, X_t], dim=1)

        for layer in self.layers:
            X_s, X_t = layer(X_s, X_t)

        pred_t = self.target_classifier(X_t)
        pred_s = self.source_classifier(X_s)

        pred_dict = {'y_pred':pred_t, 's_pred':pred_s, 'y_true':y[:,0].unsqueeze(-1), 's_true':y[:,1].unsqueeze(-1)}
        return pred_dict

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 2).to(self.device)
        self.batch_size = y.size(0)
        return X, y

class CrossConnectionLayer(nn.Module):
    def __init__(self,device,embedding_dim=64):
        super(CrossConnectionLayer, self).__init__()
        self.relu = nn.ReLU()
        self.a = torch.rand(1, requires_grad=True, device=device)
        self.W_source = nn.Linear(2*embedding_dim, 2*embedding_dim)
        self.W_target = nn.Linear(2*embedding_dim, 2*embedding_dim)
        self.W_trans = nn.Linear(2*embedding_dim, 2*embedding_dim)

    def forward(self, source_input, target_input):
        source = self.W_source(source_input)
        # source = source + self.W_trans(target_input) without == mlp++
        # source = source + self.W_trans(target_input)
        source = source + self.a * target_input
        source = self.relu(source)

        target = self.W_target(target_input)
        # target = target + self.W_trans(source_input)mlp++
        # target = target + self.W_trans(source_input)
        target = target + self.a * source_input
        target = self.relu(target)
        return source, target

class Loss_fn(nn.Module):
    def __init__(self):
        super(Loss_fn, self).__init__()

    def forward(self, inputs, reduction=None):

        y_pred = inputs['y_pred']
        y_true = inputs['y_true']

        s_pred = inputs['s_pred']
        s_true = inputs['s_true']
        # print('1:',y_true.shape,'2',y_pred.shape,'3',s_true.shape,'4',s_pred.shape)
        loss1 = F.binary_cross_entropy(y_pred,y_true)

        print('loss1:',loss1)

        loss2 = F.binary_cross_entropy(s_pred,s_true)
        print('loss2:',loss2)

        return loss1 + loss2