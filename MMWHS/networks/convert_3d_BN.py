import torch.nn as nn
import numpy as np

class AdaBN(nn.BatchNorm3d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False  # 控制是否开始计数

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3, 4), keepdims=True).detach()
        cur_var = x.var((0, 2, 3, 4), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W, D = x.shape

        new_mu, new_var = self.get_mu_var(x)  # moment更新的mu, sigma

        cur_mu = x.mean((2, 3, 4), keepdims=True)
        cur_std = x.std((2, 3, 4), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1, 1) + self.bias.view(1, C, 1, 1, 1)
        return new_x


def convert_encoder_to_target(net, norm,verbose=True,warm_n=5):
    def convert_norm(old_norm, new_norm, num_features):
        norm_layer = new_norm(num_features, warm_n).cuda()
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()  # 这里加载了旧BN的参数，只是随机初始化的，后来模型权重在后面checkpoint一起加载
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = net

    idx = 0
    for j, block in enumerate(layers):
        block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features)
        idx += 1
        block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features)
        idx += 1

    return net


def convert_decoder_to_target(net, norm, verbose=True, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features):
        norm_layer = new_norm(num_features, warm_n).cuda()
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = net

    idx = 0
    for j, block in enumerate(layers):
        block.bn = convert_norm(block.bn, norm, block.bn.num_features)

    return net

