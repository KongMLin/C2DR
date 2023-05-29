import torch
from torch import nn
from huawei_2022.fuxictr.pytorch.models import BaseModel
from huawei_2022.fuxictr.pytorch.layers import EmbeddingLayer, EmbeddingDictLayer, MLP_Layer, Dice
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy, cosine_similarity, softmax

class MVDNN(BaseModel):
    def __init__(self,
                 feature_map_u,
                 feature_map_t,
                 feature_map_s,
                 task='binary_classification',
                 embedding_dim=128,
                 layer_dim=[30000, 300, 300, 128],
                 learning_rate=1e-3,
                 **kwargs):
        super(MVDNN, self).__init__(feature_map_u,**kwargs)
        self.stage = 'pretrain'
        self.gamma = 1

        self.feature_map_u = feature_map_u
        self.feature_map_t = feature_map_t
        self.feature_map_s = feature_map_s

        self.user_feature_length = feature_map_u.input_length
        self.target_feature_length = feature_map_t.input_length
        self.source_feature_length = feature_map_s.input_length

        # print(feature_map_u.num_features,feature_map_u.feature_specs)
        # print(feature_map_u.input_length)
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim, sequence_pooling=True)
        self.embedding_layer_s = EmbeddingDictLayer(feature_map_s, embedding_dim)
        self.embedding_layer_t = EmbeddingDictLayer(feature_map_t, embedding_dim)
        layer_dim = [2 * embedding_dim] + layer_dim
        self.layers = [CrossConnectionLayer(dim=[layer_dim[i],layer_dim[i+1]], embedding_dim=embedding_dim) for i in range(len(layer_dim)-1)]
        self.layers = nn.Sequential(*self.layers)

        self.target_classifier = nn.Sequential()
        self.target_classifier.add_module('c_fc1', nn.Linear(128, 1))
        self.target_classifier.add_module('c_softmax', nn.Sigmoid())

        self.source_classifier = nn.Sequential()
        self.source_classifier.add_module('d_fc1', nn.Linear(128, 1))
        self.source_classifier.add_module('d_softmax', nn.Sigmoid())

        loss_fn = MVDNNLoss()
        self.optimizer_str = kwargs["optimizer"]
        self.learning_rate = learning_rate

        self.compile(self.optimizer_str, loss=loss_fn, lr=self.learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        if self.stage == 'pretrain':
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
        elif self.stage == 'train':
            with torch.no_grad():
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
        
        target_pos_mask = y[:,0] == 1
        target_neg_mask = y[:,0] == 0
        source_pos_mask = y[:,1] == 1

        target_pos = X_t[target_pos_mask]
        target_neg = X_s[target_neg_mask]
        source_pos = X_s[source_pos_mask]

        pred_t = self.target_classifier(X_t)
        pred_s = self.source_classifier(X_s)

        pred_dict = {'y_pred':pred_t, 's_pred':pred_s, 'y_true':y[:,0].unsqueeze(-1), 's_true':y[:,1].unsqueeze(-1), \
            'tgt_pos_feature':target_pos, 'tgt_neg_feature':target_neg, 'src_pos_feature':source_pos}
        return pred_dict

    def pretrain2train(self):
        loss_fn = Loss_fn()
        self.compile(self.optimizer_str, loss=loss_fn, lr=self.learning_rate)
        self.stage = 'train'
        self.model_to_device()

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 2).to(self.device)
        self.batch_size = y.size(0)
        return X, y

class CrossConnectionLayer(nn.Module):
    def __init__(self, dim, embedding_dim=64):
        super(CrossConnectionLayer, self).__init__()
        self.tanh = nn.Tanh()
        self.W_source = nn.Linear(dim[0], dim[1])
        self.W_target = nn.Linear(dim[0], dim[1])

    def forward(self, source_input, target_input):
        source = self.W_source(source_input)
        source = self.tanh(source)

        target = self.W_target(target_input)
        target = self.tanh(target)

        return source, target

class MVDNNLoss(nn.Module):
    def __init__(self):
        super(MVDNNLoss, self).__init__()
    
    def forward(self, inputs, reduction=None):
        tgt_pos = inputs['tgt_pos_feature']
        tgt_neg = inputs['tgt_neg_feature']
        src_pos = inputs['src_pos_feature']
        feature_dim = src_pos.size()[1]
        if len(tgt_pos) == 0:
            tgt_pos = tgt_neg[:1]
        src_repeated_neg = src_pos.unsqueeze(1).repeat(1, len(tgt_neg), 1)
        src_repeated_pos = src_pos.unsqueeze(1).repeat(1, len(tgt_pos), 1)
        tgt_neg = tgt_neg.reshape(-1, len(tgt_neg), feature_dim)
        print(tgt_pos.shape)
        tgt_pos = tgt_pos.reshape(-1, len(tgt_pos), feature_dim)

        pos_sim = cosine_similarity(src_repeated_pos, tgt_pos, dim=2)
        neg_sim = cosine_similarity(src_repeated_neg, tgt_neg, dim=2)
        all_sims = torch.cat((pos_sim, neg_sim), dim=1)

        PDQ = softmax(all_sims * 1, dim=1)
        loss = -PDQ[:, 0].log().mean()
        return loss

# prediction loss
class Loss_fn(nn.Module):
    def __init__(self):
        super(Loss_fn, self).__init__()

    def forward(self, inputs, reduction=None):
        y_pred = inputs['y_pred']
        y_true = inputs['y_true']

        s_pred = inputs['s_pred']
        s_true = inputs['s_true']

        loss1 = F.binary_cross_entropy(y_pred,y_true)
        print('loss1:',loss1)

        loss2 = F.binary_cross_entropy(s_pred,s_true)
        print('loss2:',loss2)

        return loss1 + loss2