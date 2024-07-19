"""
Implementation from https://github.com/spthermo/SDNet/blob/master/layers/adain.py
and https://github.com/HiLab-git/DAG4MIA/blob/main/code/models/networks/decoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = F.relu(out, inplace=True)
        out = self.fc3(out)
        out = F.relu(out, inplace=True)

        return out


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Decoder(nn.Module):
    def __init__(self, dim, out_channel):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain1 = AdaptiveInstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain2 = AdaptiveInstanceNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain3 = AdaptiveInstanceNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, out_channel, 3, 1, 1, bias=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.adain1(out)
        out = self.conv2(out)
        out = self.adain2(out)
        out = self.conv3(out)
        out = self.adain3(out)
        out = self.conv4(out)
        out = torch.tanh(out)
        return out

class Ada_Decoder(nn.Module):
    def __init__(self, anatomy_out_channel, z_length, out_channel):
        super(Ada_Decoder, self).__init__()
        self.dec = Decoder(anatomy_out_channel, out_channel)
        self.mlp = MLP(z_length, self.get_num_adain_params(self.dec), 256)

    def forward(self, anatomy, style):
        adain_params = self.mlp(style)  # [bs, z_length] --> [4, 48]
        self.assgin_adain_params(adain_params, self.dec)
        images = self.dec(anatomy)
        return images

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def assgin_adain_params(self, adain_params, model):
        """
        Assign the adain_params to the AdaIN layers in model
        """
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

def calc_vector_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=1)
    return feat_mean, feat_std


def calc_tensor_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_tensor_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.view(style_std.shape[0], 1, 1, 1).expand(
        size
    ) + style_mean.view(style_mean.shape[0], 1, 1, 1).expand(size)


def adaptive_instance_normalization2(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_vector_mean_std(content_feat)

    normalized_feat = (
        content_feat - content_mean.unsqueeze(-1).expand(size)
    ) / content_std.unsqueeze(-1).expand(size)
    return normalized_feat * style_std.unsqueeze(-1).expand(
        size
    ) + style_mean.unsqueeze(-1).expand(size)
