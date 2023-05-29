import torch
from torch import nn
import numpy as np
from huawei_2022.fuxictr.pytorch.models import BaseModel
from huawei_2022.fuxictr.pytorch.layers import EmbeddingDictLayer
import logging
import sys
from ...metrics import evaluate_metrics2

class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea):
        # event_K = self.event_K(emb_fea)
        # t = event_K
        # att = self.event_softmax(t)
        # his_fea = torch.sum(att * emb_fea, 1)
        # print(his_fea.shape)
        output = self.decoder(emb_fea)

        return output.squeeze(1)

class PTUPCDR(BaseModel):
    def __init__(self,
                 feature_map_u,
                 feature_map_b,
                 feature_map_a,
                 # b_hist = ['ad_click_list_v001', 'ad_click_list_v002', 'ad_click_list_v003'],
                 # a_hist = ['u_newsCatInterests',
                 #    'u_newsCatDislike', 'u_newsCatInterestsST', 'u_click_ca2_news'],
                 a_hist = ['feeds_hist'],
                 b_hist = ['hist'],
                 task='binary_classification',
                 embedding_dim=128,
                 learning_rate=1e-3,
                 meta_dim=128,
                 device=0,
                 **kwargs):
        super(PTUPCDR, self).__init__(feature_map_u, **kwargs)
        self.device = device

        self.feature_map_u = feature_map_u
        self.feature_map_a = feature_map_a
        self.feature_map_b = feature_map_b

        self.b_hist = b_hist
        self.a_hist = a_hist

        self.user_feature_length = feature_map_u.input_length
        self.b_feature_length = feature_map_b.input_length
        self.a_feature_length = feature_map_a.input_length
        self.full_length = len(self.feature_map_u.feature_specs)
        self.user_length = self.full_length - len(self.b_hist) - len(self.a_hist)
        self.a_length = len(self.a_hist)
        self.b_length = len(self.b_hist)
        self.embedding_dim = embedding_dim
        self.embedding_layer_u = EmbeddingDictLayer(feature_map_u, embedding_dim)
        self.embedding_layer_a = EmbeddingDictLayer(feature_map_a, embedding_dim)
        self.embedding_layer_b = EmbeddingDictLayer(feature_map_b, embedding_dim)

        self.src_model = nn.Linear(embedding_dim, embedding_dim)
        self.tgt_model = nn.Linear(embedding_dim, embedding_dim)
        self.aug_model = nn.Linear(embedding_dim, embedding_dim)
        self.meta_net = MetaNet(embedding_dim, meta_dim)
        self.mapping = nn.Linear(embedding_dim, embedding_dim, False)
        self.stage = 'train_src'
        self.optimizer_src = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer_tgt = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer_meta = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer_aug = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer_map = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer = [self.optimizer_src,self.optimizer_tgt,self.optimizer_meta,self.optimizer_aug,self.optimizer_map]
        self.loss_fn = nn.MSELoss()
        # self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)

        if self.stage == 'train_src':
            source_mask = [y[:, -1] == 0]
            x, y = X[source_mask], y[source_mask]
            if y.shape[0] == 0:
                return None
            X_u = self.embedding_layer_u(x[:, :self.user_feature_length])
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
            X_u = torch.mean(X_u,dim=1)
            I_a = self.embedding_layer_a(x[:, self.user_feature_length:self.user_feature_length + self.a_feature_length])
            I_a = self.embedding_layer_a.dict2tensor(I_a)
            I_a = I_a.sum(dim=1) / self.a_feature_length

            x = torch.sum(self.src_model(X_u) * I_a, dim=1)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret[self.stage] = x
        elif self.stage in ['train_tgt', 'test_tgt']:
            target_mask = [y[:, -1] == 1]
            x, y = X[target_mask], y[target_mask]
            if y.shape[0] == 0:
                return None

            X_u = self.embedding_layer_u(x[:, :self.user_feature_length])
            X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
            X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length
            for feature_name, feature in X_u.items():
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
            X_u = torch.mean(X_u, dim=1)
            I_b = self.embedding_layer_b(x[:, self.user_feature_length + self.a_feature_length:])
            I_b = self.embedding_layer_b.dict2tensor(I_b)
            I_b = I_b.sum(dim=1) / self.b_feature_length

            x = torch.sum(self.tgt_model(X_u) * I_b, dim=1)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret[self.stage] = x
        elif self.stage in ['train_aug', 'test_aug']:
            X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
            X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
            X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length
            for feature_name, feature in X_u.items():
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
            X_u = torch.mean(X_u, dim=1)
            I_a = self.embedding_layer_a(X[:, self.user_feature_length:self.user_feature_length + self.a_feature_length])
            I_a = self.embedding_layer_a.dict2tensor(I_a)
            I_a = I_a.sum(dim=1) / self.a_feature_length

            I_b = self.embedding_layer_b(X[:, self.user_feature_length + self.a_feature_length:])
            I_b = self.embedding_layer_b.dict2tensor(I_b)
            I_b = I_b.sum(dim=1) / self.b_feature_length

            x = torch.sum(self.tgt_model(X_u) * I_a, dim=1) + torch.sum(self.tgt_model(X_u) * I_b, dim=1)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret[self.stage] = x
        elif self.stage in ['test_meta', 'train_meta']:
            I_b = self.embedding_layer_b(X[:, self.user_feature_length + self.a_feature_length:])
            I_b = self.embedding_layer_b.dict2tensor(I_b)
            I_b = I_b.sum(dim=1) / self.b_feature_length

            X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
            X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
            X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length
            for feature_name, feature in X_u.items():
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
            X_u = torch.mean(X_u, dim=1)

            I_a = self.embedding_layer_a(X[:, self.user_feature_length:self.user_feature_length + self.a_feature_length])
            I_a = self.embedding_layer_a.dict2tensor(I_a)
            I_a = I_a.sum(dim=1) / self.a_feature_length
            mapping = self.meta_net.forward(I_a).view(-1, self.embedding_dim, self.embedding_dim)
            # print(X_u.shape,mapping.shape)
            uid_emb = torch.bmm(X_u.unsqueeze(1), mapping).squeeze(1)
            # emb = torch.cat([uid_emb, I_b], 1)
            output = torch.sum(uid_emb * I_b, 1)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret[self.stage] = output
        elif self.stage == 'train_map':
            X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
            X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
            X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length
            for feature_name, feature in X_u.items():
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
            X_u = torch.mean(X_u, dim=1)

            src_emb = self.src_model(X_u)
            src_emb = self.mapping.forward(src_emb)

            tgt_emb = self.tgt_model(X_u)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret['src_emb'] = src_emb
            ret['tgt_emb'] = tgt_emb
        elif self.stage == 'test_map':
            X_u = self.embedding_layer_u(X[:, :self.user_feature_length])
            X_u_emb = self.embedding_layer_u.dict2tensor(X_u, user_feature=self.user_length)
            X_user = X_u_emb[:, :self.user_length].sum(dim=1) / self.user_length
            for feature_name, feature in X_u.items():
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
            X_u = torch.mean(X_u, dim=1)
            uid_emb = self.mapping.forward(self.src_model(X_u))

            I_b = self.embedding_layer_b(X[:, self.user_feature_length + self.a_feature_length:])
            I_b = self.embedding_layer_b.dict2tensor(I_b)
            I_b = I_b.sum(dim=1) / self.b_feature_length
            # emb = torch.cat([uid_emb, I_b], 1)
            x = torch.sum(uid_emb * I_b, 1)
            ret = {'label': y[:,0], 'flabel': y[:,1], 'user_id': X[:,0]}
            ret[self.stage] = x
        return ret

    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 3).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    def evaluate_generator(self, data_generator):
        self.eval() # set to evaluation mode
        self.stage = 'test_map'
        with torch.no_grad():
            pred = []
            # pred_b = []
            y_true = []
            # yb_true = []
            user_list = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)

                pred.extend(return_dict[self.stage].data.cpu().numpy().reshape(-1))
                # pred_b.extend(return_dict["pred_b"].data.cpu().numpy().reshape(-1))
                y_true.extend(return_dict["label"].data.cpu().numpy().reshape(-1))
                # yb_true.extend(return_dict["yb_true"].data.cpu().numpy().reshape(-1))
                user_list.extend(return_dict["user_id"].data.cpu().numpy().reshape(-1))

            pred = np.array(pred, np.float64)
            # pred_b = np.array(pred_b, np.float64)
            # print(pred_a.shape,pred_b.shape)
            y_true = np.array(y_true, np.float64)
            # yb_true = np.array(yb_true, np.float64)
            user_list = np.array(user_list, np.float64)

            # yb_true[0] = 1

            val_logs = self.evaluate_metrics([user_list, y_true, pred], self._validation_metrics)
            # val_logs2 =  self.evaluate_metrics([user_list, yb_true, pred_b], self._validation_metrics)

            return val_logs,  pred, y_true

    def evaluate_metrics(self, return_list, metrics):
        return evaluate_metrics2(return_list, metrics)

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
        self.stage='train_tgt'
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch, 'test_tgt')
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage1 finished")

        self.stage = 'train_aug'
        logging.info("Start training stage2: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch, 'test_aug')
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage2 finished")

        self.stage = 'train_src'
        logging.info("Start training stage3: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage3 finished")

        self.stage = 'train_map'
        logging.info("Start training stage4: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch, 'test_map')
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage4 finished")

        #注释掉下面的内容就是EMCDR
        self.stage = 'train_meta'
        logging.info("Start training stage5: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch, 'test_meta')
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training stage5 finished")

        logging.info("Training finished.")

    def train_one_epoch(self, data_generator, epoch, stage=None):
        epoch_loss = 0
        stage_dict = {'train_tgt':1,'train_aug':2,
                      'train_src':3,'train_map':4,
                      'train_meta':5}
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer[stage_dict[self.stage]-1].zero_grad()
            return_dict = self.forward(batch_data)
            label_name = 'label' if self.stage[-3:] == 'tgt' else 'flabel'
            if stage_dict[self.stage] == 4:
                loss = self.loss_fn(return_dict['src_emb'], return_dict['tgt_emb'])
            else:

                loss = self.loss_fn(return_dict[self.stage],return_dict[label_name])
            # loss = self.loss_fn(return_dict[])
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer[stage_dict[self.stage]-1].step()
            epoch_loss += loss.item()
            self.on_batch_end(batch_index,stage)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def on_batch_end(self, batch,stage, logs={}):
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches_per_epoch == 0:
            if stage:
               self.stage = stage
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            val_logs, y_pred, y_true = self.evaluate_generator(self.valid_gen)
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch + 1, self._batches_per_epoch))
