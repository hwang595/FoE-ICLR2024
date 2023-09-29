'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGGCustom': [64, 'M', 128, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class MCDropoutVGG(nn.Module):
    def __init__(self, vgg_name):
        super(MCDropoutVGG, self).__init__()
        self.avg_layer_num = 3
        self.avg_features, self.personalized_layers = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10))

    def forward(self, x):
        out = self.avg_features(x)
        out = self.personalized_layers(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        avg_layers = []
        personalized_layers = []
        in_channels = 3
        avg_layer_counter = 0
        for x in cfg:
            if x == 'M':
                if avg_layer_counter < self.avg_layer_num:
                    avg_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    personalized_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if avg_layer_counter < self.avg_layer_num:
                    avg_layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(x),
                                nn.ReLU(inplace=True)]
                else:
                    personalized_layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]                    
                in_channels = x
                avg_layer_counter += 1
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        personalized_layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*avg_layers), nn.Sequential(*personalized_layers)


class GaussianVGG(nn.Module):
    def __init__(self, vgg_name):
        super(GaussianVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        #self.var_features = self._make_layers(cfg[vgg_name])
        self.var_classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out_featmap = out.view(out.size(0), -1)
        out = self.classifier(out_featmap)

        #out_var = self.var_features(x)
        #out_var = out_var.view(out_var.size(0), -1)
        #out_var = self.var_classifier(out_var) ** 2
        out_var = self.var_classifier(out_featmap) ** 2
        return F.log_softmax(out, dim=1), out_var

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class FactorizedVGG(nn.Module):
    def __init__(self, vgg_name):
        super(FactorizedVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if x_index == 0:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, int(x/16), kernel_size=3, padding=1),
                               nn.BatchNorm2d(int(x/16)),
                               nn.Conv2d(int(x/16), x, kernel_size=1),
                               nn.ReLU(inplace=True)]                    
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGGFeatureMap(nn.Module):
    def __init__(self, vgg_name):
        super(VGGFeatureMap, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.out_put_dim = 512

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
