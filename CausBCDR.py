import torch
from torch import nn
import numpy as np
from huawei_2022.fuxictr.pytorch.models import BaseModel
from huawei_2022.fuxictr.pytorch.layers import EmbeddingDictLayer
import logging
import sys
import torch.nn.functional as F
from torch.autograd import Function
from ..torch_utils import get_optimizer
from ...metrics import evaluate_metrics2

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CausBCDR(BaseModel):
    def __init__(self,
                 feature_map_u,
                 feature_map_b,
                 feature_map_a,
                 # b_hist = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003'],
                 # a_hist = ['u_newsCatInterests', 'u_newsCatDislike', 'u_click_ca2_news','u_newsCatInterestsST'],
                 task='binary_classification',
                 b_hist = ['movie_hist'],
                 a_hist = ['book_hist'],
                 # b_hist = ['hist'],
                 # a_hist = ['feeds_hist'],
                 embedding_dim=128,
                 dnn_hidden_units=[128],
                 net_dropout=0,
                 batch_norm=False,
                 learning_rate1=1e-3,
                 learning_rate2=1e-4,
                learning_rate3 = 1e-2,
                 tensorboard = None,
                 **kwargs):
        super(CausBCDR, self).__init__(feature_map_u,**kwargs)

        self.seq_agg = kwargs['seq_agg']
        self.interset_attn = kwargs['interest_attn']
        self.bi_pooling = kwargs['bi_pooling']
        self.dnn = kwargs['dnn']
        self.batch_size = kwargs['batch_size']
        self.domain_ag = kwargs['domain_agnostic']
        self.domain_pred = kwargs['domain_pred']
        self.c_domain_pred = kwargs['c_domain_pred']
        self.tensorboard = tensorboard
        self.optimizer = None
        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1
        #amazon dataset
        # target_hist = ['hist']
        # source_hist = ['feeds_hist']
        self.feature_map_u = feature_map_u
        self.feature_map_a = feature_map_a
        self.feature_map_b = feature_map_b

        self.a_hist = a_hist
        self.b_hist = b_hist

        self.weight = nn.Parameter(torch.randn(self.batch_size))

        self.user_feature_length = feature_map_u.input_length
        self.a_feature_length = feature_map_a.input_length
        self.b_feature_length = feature_map_b.input_length

        self.full_length = len(feature_map_u.feature_specs)
        self.user_length = self.full_length - len(b_hist) - len(a_hist)
        self.a_length = len(a_hist)
        self.b_length = len(b_hist)
        # print(feature_map_u.num_features,feature_map_u.feature_specs)
        # print(feature_map_u.input_length)
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim)
        self.embedding_layer_a = EmbeddingDictLayer(feature_map_a, embedding_dim)
        self.embedding_layer_b = EmbeddingDictLayer(feature_map_b, embedding_dim)
        self.encoder_itemA = Encoder(embedding_dim,embedding_dim)
        self.encoder_itemB = Encoder(embedding_dim,embedding_dim)
        self.encoder_userA = Encoder(embedding_dim,embedding_dim)
        self.encoder_userB = Encoder(embedding_dim,embedding_dim)
        self.encoder_s = Encoder(embedding_dim,embedding_dim)
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(embedding_dim, 1))
        self.domain_classifier.add_module('d_softmax', nn.Sigmoid())
        self.attention_layers_item_A = ItemAttentionLayer(embedding_dim=embedding_dim,device=kwargs['gpu'])
        self.attention_layers_item_B = ItemAttentionLayer(embedding_dim=embedding_dim,device=kwargs['gpu'])
        self.attn_net_a = AttentionLayer(embedding_dim)
        self.attn_net_b = AttentionLayer(embedding_dim)
        self.task_a = Task(2*embedding_dim)
        self.task_b = Task(2*embedding_dim)

        loss_fn1 = Loss_fn1(domain_pred=self.domain_pred,c_domain_pred=self.c_domain_pred)
        self.compile(kwargs["optimizer"], loss=loss_fn1, lr=learning_rate1)
        loss_fn2 = Loss_fn2(domain_pred=self.domain_pred,c_domain_pred=self.c_domain_pred)
        self.compile(kwargs["optimizer"], loss=loss_fn2, lr=learning_rate2)
        loss_fn3 = Loss_fn3(domain_pred=self.domain_pred,c_domain_pred=self.c_domain_pred)
        self.compile(kwargs["optimizer"], loss=loss_fn3, lr=learning_rate3)

        self.reset_parameters()
        self.model_to_device()
        self.stage = 'train1'



    def forward(self,inputs):
        if self.training:
            if self.p < 1:
                self.p += 1 / 15000
                self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        X, y = self.inputs_to_device(inputs)
        # y[:,1] = (y[:,1] + 1) / 2
        emb_itemA = self.embedding_layer_a(X[:, self.user_feature_length:self.user_feature_length + self.a_feature_length])
        emb_itemA = self.embedding_layer_a.dict2tensor(emb_itemA)
        emb_itemA = emb_itemA.sum(dim=1) / self.a_feature_length

        # X_u = X[:, :self.user_feature_length]
        X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
        # X_u = self.embedding_layer_u.dict2tensor(X_u)

        emb_itemB = self.embedding_layer_b(X[:, self.user_feature_length + self.a_feature_length:])
        emb_itemB = self.embedding_layer_b.dict2tensor(emb_itemB)
        emb_itemB = emb_itemB.sum(dim=1) / self.b_feature_length

        X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)

        X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length

        for feature_name, feature in X_u.items():
            # print(feature_name,feature.shape)
            if feature_name in self.b_hist:
                if not self.seq_agg:
                    tmp_feature = torch.mean(feature, dim=1)
                else:
                    tmp_feature = self.attention_layers_item_B(feature, emb_itemB, X_user)
                X_u[feature_name] = tmp_feature

            if feature_name in self.a_hist:
                if not self.seq_agg:
                    tmp_feature = torch.mean(feature, dim=1)
                else:
                    tmp_feature = self.attention_layers_item_A(feature, emb_itemA, X_user)
                X_u[feature_name] = tmp_feature

        X_u = self.embedding_layer_u.dict2tensor(X_u)
        X_a_hist = X_u[:, self.user_length:self.user_length + self.a_length].sum(dim=1) / self.a_length
        X_b_hist = X_u[:, self.user_length + self.a_length:].sum(dim=1) / self.b_length
        X_u = torch.cat([X_user.unsqueeze(1), X_a_hist.unsqueeze(1), X_b_hist.unsqueeze(1)], dim=1)
        X_u = torch.mean(X_u,dim=1)
        X_itemA = self.encoder_itemA(emb_itemA)
        X_itemB = self.encoder_itemB(emb_itemB)

        X_A = self.encoder_userA(X_u)
        X_B = self.encoder_userB(X_u)

        X = self.encoder_s(X_u)

        pred_d_X = self.domain_classifier(ReverseLayerF.apply(X,self.alpha))
        pred_d_XA = self.domain_classifier(ReverseLayerF.apply(X_A,self.alpha))
        pred_d_XB = self.domain_classifier(ReverseLayerF.apply(X_B,self.alpha))

        # h_A = self.attn_net_a(X_A,X,X_B.detach())
        # h_B = self.attn_net_b(X_A.detach(),X,X_B)
        tmp_a = X + X_A
        tmp_b = X + X_B
        # h_A = torch.mean([X,X_A],keepdim=True)
        # h_B = torch.mean([X,X_B],keepdim=True)
        pred_a = self.task_a(torch.concat([tmp_a, X_itemA],dim=1))
        pred_b = self.task_b(torch.concat([tmp_b, X_itemB], dim=1))


        pred_sa, pred_sb = self.forward_aux(X,emb_itemA,emb_itemB)

        pred_dict = {'pred_a':pred_a,'pred_b':pred_b,'pred_d_X':pred_d_X,'pred_d_XA':pred_d_XA,'pred_d_XB':pred_d_XB,
                     'X':X,'X_A':X_A,'X_B':X_B,'pred_sa':pred_sa,'pred_sb':pred_sb,
                     'ya_true':y[:,0].unsqueeze(-1), 'yb_true':y[:,1].unsqueeze(-1),
                     'weight':self.weight,'user_id': X[:,0]
                     }

        return pred_dict

    def compile(self, optimizer, loss, lr):
        if self.optimizer is None:
            self.optimizer = []
            self.loss_fn = []
        self.optimizer.append(get_optimizer(optimizer, self.parameters(), lr))
        # self.loss_fn = get_loss_fn(loss)
        self.loss_fn.append(loss)



    def forward_aux(self, X, emb_itemA, emb_itemB):
        if not self.training:
            return None,None
        self.encoder_itemA.eval()
        self.encoder_itemB.eval()
        self.task_a.eval()
        self.task_b.eval()
        X_itemA = self.encoder_itemA(emb_itemA)
        X_itemB = self.encoder_itemB(emb_itemB)
        pred_sa = self.task_a(torch.concat([X,X_itemA],dim=1))
        pred_sb = self.task_b(torch.concat([X,X_itemB], dim=1))


        return pred_sa,pred_sb

    def evaluate_generator(self, data_generator):
        self.eval() # set to evaluation mode
        with torch.no_grad():

            pred_a = []
            pred_b = []
            ya_true = []
            yb_true = []
            user_list = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)

                pred_a.extend(return_dict["pred_a"].data.cpu().numpy().reshape(-1))
                pred_b.extend(return_dict["pred_b"].data.cpu().numpy().reshape(-1))
                ya_true.extend(return_dict["ya_true"].data.cpu().numpy().reshape(-1))
                yb_true.extend(return_dict["yb_true"].data.cpu().numpy().reshape(-1))
                user_list.extend(return_dict['user_id'].data.cpu().numpy().reshape(-1))
            pred_a = np.array(pred_a, np.float64)
            pred_b = np.array(pred_b, np.float64)
            # print(pred_a.shape,pred_b.shape)
            ya_true = np.array(ya_true, np.float64)
            yb_true = np.array(yb_true, np.float64)
            user_list = np.array(user_list,np.float64)


            ya_true[0] = 1
            yb_true[0] = 1
            # if self.tensorboard is not None:
            #     self.tensorboard = self.tensorboard.append(
            #         pd.DataFrame({'v_u': v_u, 'v_s': v_s, 'v_t': v_t,'y_true':y_true}), ignore_index=True)
                # self.tensorboard['v_u'] = v_u1
                # self.tensorboard['v_s'] = v_s1
                # self.tensorboard['v_t'] = v_t1
            val_logs = self.evaluate_metrics([user_list,ya_true,pred_a],self._validation_metrics)
            print((yb_true==1).sum())
            print((yb_true==0).sum())
            val_logs2 =  self.evaluate_metrics([user_list,yb_true,pred_b], self._validation_metrics)

            # print(y_pred_source.shape,y_true_source.shape,y_pred_target.shape,y_true_target.shape)
            # auc_source = self.evaluate_metrics(y_true_source,y_pred_source, self._validation_metrics)['AUC']
            # auc_target = self.evaluate_metrics(y_true_target,y_pred_target, self._validation_metrics)['AUC']
            # A_distance = 2 * abs(auc_source - auc_target)
            # print(A_distance)
            return val_logs, val_logs2, pred_a, ya_true, pred_b, yb_true

    def evaluate_metrics(self, return_list, metrics):
        return evaluate_metrics2(return_list, metrics)

    def add_loss(self, inputs, reduction="mean"):
        return_dict = self.forward(inputs)
        loss, loss_list = self.loss_fn(return_dict, reduction=reduction)
        return loss, loss_list

    def get_total_loss(self, inputs):
        total_loss, loss_list = self.add_loss(inputs)
        total_loss = total_loss + self.add_regularization()
        return total_loss, loss_list

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 2).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    def fit_generator(self, data_generator, epochs=3, validation_data=None,
                      verbose=0, max_gradient_norm=10., **kwargs):
        """
        Training a model and valid accuracy.
        Inputs:
        - iter_train: I
        - iter_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)
        self._every_x_batches = int(np.ceil(self._every_x_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose

        logging.info("Start training stage1: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            # if self._stop_training:
            #     break
            # else:
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage1 finished")

        self.stage = 'train2'
        logging.info("Start training stage2: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            # if self._stop_training:
            #     break
            # else:
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage2 finished")

        self.eval()
        self.attn_net_a.train()
        self.attn_net_b.train()
        self.task_a.train()
        self.task_b.train()
        self.stage = 'train3'
        logging.info("Start training stage3: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            # if self._stop_training:
            #     break
            # else:
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage3 finished")

        logging.info("Training finished.")


    def train_one_epoch(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        niter = 1
        if self.stage[-1] == '3' or self.stage[-1] == '1' :

            niter = 2
        for _ in range(niter):
            batch_iterator = data_generator
            if self._verbose > 0:
                from tqdm import tqdm
                batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_index, batch_data in enumerate(batch_iterator):
                self.optimizer[int(self.stage[-1])-1].zero_grad()
                return_dict = self.forward(batch_data)
                loss, loss_list = self.loss_fn[int(self.stage[-1])-1](return_dict)
                if self.stage[-1] == '3':
                    parm = {}
                    for name, parameters in self.named_parameters():
                        parm[name] = parameters
                    w_a = parm['task_a.linear.weight']
                    w_b = parm['task_b.linear.weight']
                    grad_a = w_a / (w_a.norm(2, dim=1, keepdim=True) + 1e-9)
                    grad_b = w_b / (w_b.norm(2, dim=1, keepdim=True) + 1e-9)
                    bs, length = grad_b.shape
                    grad_a = grad_a.reshape(bs, 1, length)
                    grad_b = grad_b.reshape(bs, length, 1)
                    loss_grad_orthogonal = torch.square(torch.bmm(grad_a, grad_b).cuda()).sum()
                    loss = loss + loss_grad_orthogonal
                    print('loss_grad_orthogonal: ',loss_grad_orthogonal)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
                self.optimizer[int(self.stage[-1])-1].step()
                epoch_loss += loss.item()
                self.on_batch_end(batch_index)
                if self._stop_training:
                    break
        return epoch_loss / self._batches_per_epoch


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



class AttentionLayer(nn.Module):
    def __init__(self,embedding_dim=64):
        super(AttentionLayer, self).__init__()
        self.relu = nn.ReLU()
        self.Linear_V_S = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Linear_G_S = nn.Linear(embedding_dim, 1)
        self.Linear_V_A = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Linear_G_A = nn.Linear(embedding_dim, 1)
        self.Linear_V_B = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Linear_G_B = nn.Linear(embedding_dim, 1)
        self.Linear_smooth = nn.Linear(embedding_dim*3,embedding_dim)
    def forward(self, X_A,X,X_B):
        v_S = torch.exp(self.Linear_G_S(self.relu(self.Linear_V_S(X))))
        v_A = torch.exp(self.Linear_G_A(self.relu(self.Linear_V_A(X_A))))
        v_B = torch.exp(self.Linear_G_B(self.relu(self.Linear_V_B(X_B))))
        tmp = self. Linear_smooth(torch.cat([v_S * X, v_A * X_A, v_B * X_B], dim=1))
        return tmp

class Loss_fn1(nn.Module):
    def __init__(self, classes=2, alpha=1,beta=10000,gamma=1,eta=0.5,delta=1,epsilon=1,
                 domain_pred=True,
                 c_domain_pred=True):

        super(Loss_fn1,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.delta = delta
        self.epsilon = epsilon
        # self.fn = nn.MSELoss()
        # self.fn1 = F.binary_cross_entropy()
        # self.fn2 = F.binary_cross_entropy()


    def forward(self,input,reduction=None):
        pred_a = input['pred_a']
        pred_b = input['pred_b']
        pred_d_X = input['pred_d_X']
        pred_d_XA = input['pred_d_XA']
        pred_d_XB = input['pred_d_XB']
        X = input['X']
        X_A = input['X_A']
        X_B = input['X_B']
        pred_sa = input['pred_sa']
        pred_sb = input['pred_sb']
        ya_true = input['ya_true']
        yb_true = input['yb_true']
        weight = input['weight']
        emb_dim = len(X[0])

        loss_s = F.binary_cross_entropy(pred_sa,ya_true) + F.binary_cross_entropy(pred_sb,yb_true)
        weight = F.softmax(weight.detach()).unsqueeze(-1)
        # loss_a = F.binary_cross_entropy(pred_a,ya_true,weight=weight.data)
        # loss_b = F.binary_cross_entropy(pred_b, yb_true,weight=weight.data)
        loss_a = F.binary_cross_entropy(pred_a, ya_true)
        loss_b = F.binary_cross_entropy(pred_b, yb_true)
        loss_confusion = 2 * F.binary_cross_entropy(pred_d_X, torch.ones_like(pred_d_X))\
                         + F.binary_cross_entropy(pred_d_XA, torch.zeros_like(pred_d_XA)) \
                         + F.binary_cross_entropy(pred_d_XB, torch.zeros_like(pred_d_XB))
        loss_vec_orthogonal_tmp = F.cosine_similarity(X,X_A) + F.cosine_similarity(X_B,X_A) + F.cosine_similarity(X,X_B)
        loss_vec_orthogonal = loss_vec_orthogonal_tmp.mean(dim=0)
        X = X.unsqueeze(0)
        X_A = X_A.unsqueeze(0)
        X_B = X_B.unsqueeze(0)
        def vec_mse_loss(X1,X2):
            vec1 = torch.matmul(torch.matmul(X1.transpose(2, 1), torch.diag(weight.squeeze(-1)).unsqueeze(0)), X2) / len(weight)
            # print(weight.shape)
            vec2 = torch.matmul(X1.transpose(2, 1), weight.repeat([1,emb_dim]).unsqueeze(0)) \
                    / len(weight) * torch.matmul(X2.transpose(2, 1), weight.unsqueeze(0)) / len(weight)
            return F.mse_loss(vec1.squeeze(0), vec2.squeeze(0))


        X = X.squeeze(0)
        X_A = X_A.squeeze(0)
        X_B = X_B.squeeze(0)
        # grad_a = torch.autograd.backward(loss_a,X)
        # grad_b = torch.autograd.backward(loss_b,X)
        # loss_grad_orthogonal = F.cosine_similarity(grad_a,grad_b)
        total_loss = self.alpha*loss_s + loss_a + loss_b + self.eta*loss_confusion+self.beta*loss_vec_orthogonal
        loss_list = [ loss_s, loss_a,loss_b,loss_confusion]
        print('loss_s: ', loss_s, 'loss_a: ', loss_a,'loss_b: ', '\n',
              loss_b,'loss_confusion: ',loss_confusion ,'loss_vec:',loss_vec_orthogonal)
        return total_loss, loss_list

class Loss_fn2(nn.Module):
    def __init__(self, classes=2, alpha=1,beta=1,gamma=1,eta=1,delta=1,epsilon=1,
                 domain_pred=True,
                 c_domain_pred=True):
        super(Loss_fn2, self).__init__()
        self.alpha = alpha
    def forward(self,input,reduction=None):
        X = input['X']
        X_A = input['X_A']
        X_B = input['X_B']
        weight = input['weight']
        emb_dim = len(X[0])
        weight = F.softmax(weight.detach()).unsqueeze(-1)
        X = X.unsqueeze(0)
        X_A = X_A.unsqueeze(0)
        X_B = X_B.unsqueeze(0)
        def vec_mse_loss(X1,X2):
            # print(X1.shape)
            vec1 = torch.matmul(torch.matmul(X1.transpose(2, 1), torch.diag(weight.squeeze(-1)).unsqueeze(0)), X2) / len(weight)
            # print(weight.shape)
            vec2 = torch.matmul(X1.transpose(2, 1), weight.repeat([1,emb_dim]).unsqueeze(0)) \
                    / len(weight) * torch.matmul(X2.transpose(2, 1), weight.unsqueeze(0)) / len(weight)
            return F.mse_loss(vec1.squeeze(0), vec2.squeeze(0))
        loss_vec_orthogonal = vec_mse_loss(X,X_A) + vec_mse_loss(X_B,X_A) + vec_mse_loss(X,X_B)
        total_loss = self.alpha * loss_vec_orthogonal
        loss_list = [loss_vec_orthogonal]
        return total_loss, loss_list

class Loss_fn3(nn.Module):
    def __init__(self, classes=2, alpha=1,beta=1,gamma=1,eta=1,delta=1,epsilon=1,
                 domain_pred=True,
                 c_domain_pred=True):
        super(Loss_fn3, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, reduction=None):
        pred_a = input['pred_a']
        pred_b = input['pred_b']
        pred_d_X = input['pred_d_X']
        pred_d_XA = input['pred_d_XA']
        pred_d_XB = input['pred_d_XB']
        X = input['X']
        X_A = input['X_A']
        X_B = input['X_B']
        pred_sa = input['pred_sa']
        pred_sb = input['pred_sb']
        ya_true = input['ya_true']
        yb_true = input['yb_true']
        weight = input['weight']
        # print(weight)
        emb_dim = len(X[0])

        weight = F.softmax(weight.detach()).unsqueeze(-1)
        # print(weight)
        loss_a = F.binary_cross_entropy(pred_a,ya_true,weight=weight.data)
        loss_b = F.binary_cross_entropy(pred_b, yb_true,weight=weight.data)
        def vec_mse_loss(X1, X2):
            vec1 = torch.matmul(torch.matmul(X1.transpose(2, 1), torch.diag(weight.squeeze(-1)).unsqueeze(0)),
                                X2) / len(weight)
            # print(weight.shape)
            vec2 = torch.matmul(X1.transpose(2, 1), weight.repeat([1, emb_dim]).unsqueeze(0)) \
                   / len(weight) * torch.matmul(X2.transpose(2, 1), weight.unsqueeze(0)) / len(weight)
            return F.mse_loss(vec1.squeeze(0), vec2.squeeze(0))


        # grad_a = torch.autograd.backward(loss_a,X)
        # grad_b = torch.autograd.backward(loss_b,X)
        # loss_grad_orthogonal = F.cosine_similarity(grad_a,grad_b)

        total_loss =  self.alpha * loss_a + self.beta * loss_b
        loss_list = [ loss_a, loss_b]
        print('loss_a: ',loss_a,'loss_b: ', loss_b)
        return total_loss, loss_list




class Encoder(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Encoder,self).__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
    def forward(self,input):
        return self.relu(self.linear2(self.relu(self.linear(input))))


class Task(nn.Module):
    def __init__(self,input_dim):
        super(Task,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.linear2 = nn.Linear(input_dim, input_dim, bias=False)
    def forward(self, input):
        return self.sigmoid(self.linear(self.linear2(input)))

