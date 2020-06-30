import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'VGG_GAP_GAS', 'vgg16', 'vgg_gap_gas',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

import torch.nn.functional as F
from .. import misc as ms
from . import base_model as bm

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


class FeatureExtracter_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        
        backbone = vgg16(pretrained=True)

        self.tmp = list(backbone.features.children());
        for i in range(8) :
            self.tmp.pop();

        # replace relu layers with prelu
        self.replace_relu_with_prelu();

        self.features_1 = nn.Sequential(*self.tmp);

    def replace_relu_with_prelu(self) :
        id_relu = [1,3,6,8,11,13,15,18,20,22];
        for i in id_relu :
            self.tmp[i] = nn.PReLU(self.tmp[i-1].out_channels);

    
    def extract_features(self, x_input):
        return self.features_1(x_input)
from skimage.segmentation import felzenszwalb, slic, quickshift
class GAM(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set)
        backbone = vgg16(pretrained=True)

        self.tmp = list(backbone.features.children());
        for i in range(8) :
            self.tmp.pop();

        # replace relu layers with prelu
        self.replace_relu_with_prelu();

        self.features_1 = nn.Sequential(*self.tmp);
        # self.features_2 = nn.AvgPool2d(kernel_size=input_size, stride=input_size); # 45x80
        self.classifier = nn.Linear(in_features=512, out_features=1);

    def replace_relu_with_prelu(self) :
        id_relu = [1,3,6,8,11,13,15,18,20,22];
        for i in id_relu :
            self.tmp[i] = nn.PReLU(self.tmp[i-1].out_channels);


    def forward(self, x):
        
        x = self.features_1(x);
        x = x.abs();
        # count = self.features_2(x);
        input_size = (x.size(2), x.size(3))


        count = F.avg_pool2d(x, kernel_size=input_size, stride=input_size)
        count = count.view(count.size(0), -1);
        count = self.classifier(count);

        shape = x.shape[-2:]

        x = x.view(x.size(0), x.size(1), -1);
        x = x.mul(self.classifier.weight.data.unsqueeze(2));

        x = x.sum(1);
        x = x.abs();
        max_, _ = x.data.max(1);
        x.data.div_(max_.unsqueeze(1).expand_as(x));

        return {"count":count, "cam":x.reshape(shape)}

    @torch.no_grad()
    def predict(self, batch, **options):
        self.eval()

        n,c,h,w = batch["images"].shape
   

        pred_dict = self(batch["images"].cuda())

        
        return {"counts":ms.t2n(pred_dict["count"]).ravel()}

    def get_points(self, batch):
        from skimage.segmentation import mark_boundaries
        if 1:
            img = ms.f2l(ms.t2n((ms.denormalize(batch["images"])))).squeeze()
            pred_dict = self(batch["images"].cuda())
            h, w = batch["images"].shape[-2:]

            mask = ms.t2n(F.interpolate(pred_dict["cam"][None][None], 
                                size=(h,w))).squeeze()

            segments_slic = slic(img, n_segments=1000, compactness=5, sigma=0.5)
            labels = np.unique(segments_slic)
            scoreList = np.zeros(len(labels))
            for i, l in enumerate(labels):
                scoreList[i] = (mask*(segments_slic==l)).mean()

            selected = np.argsort(scoreList)[-batch["counts"].item():]
            points = np.zeros(segments_slic.shape)
            for s in selected:
                ind = segments_slic == s
                max_val = mask[ind].max()

                points[mask == max_val] = 1

            return points
            # ms.images(batch["images"], points.astype(int),win="points",
            #             denorm=1,enlarge=1)
            # ms.images(mark_boundaries(img, segments_slic))


    @torch.no_grad()
    def visualize(self, batch, **options):
        pred_dict = self(batch["images"].cuda())
        h, w = batch["images"].shape[-2:]
        mask = ms.t2n(F.interpolate(pred_dict["cam"][None][None], 
                            size=(h,w))).squeeze()
        ms.images(ms.gray2cmap(mask), win="mask")

        ms.images(batch["images"], mask>0.5, denorm=1)

        return mask

from ..models import gans
from scipy.ndimage.filters import gaussian_filter 
import numpy as np

from scipy import fftpack

def kernel(h, w, sigma=1.5):
    from scipy import stats

    sx, sy = h, w
    X, Y = np.ogrid[0:sx, 0:sy]
    psf = stats.norm.pdf(np.sqrt((X - sx/2)**2 + (Y - sy/2)**2), 0, sigma)    
    psf /= psf.sum()

    return psf

def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel array."""
#     import scipy.stats as st

#     interval = (2*nsig+1.)/(kernlen)
#     x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw/kernel_raw.sum()
#     return kernel

# if 1:
#     # First a 1-D  Gaussian
#     t = np.linspace(-10, 10, 30)
#     bump = np.exp(-0.1*t**2)
#     bump /= np.trapz(bump) # normalize the integral to 1

#     # make a 2-D kernel out of it
#     kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
#     img = points
#     kernel_ft = fftpack.fft2(gkern(kernlen=21, nsig=3), shape=img.shape[:2], axes=(0, 1))

#     # convolve
#     img_ft = fftpack.fft2(img, axes=(0, 1))[:, :, np.newaxis]
#     # the 'newaxis' is to match to color direction
#     img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
#     img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

#     # clip values to range
#     img2 = np.clip(img2, 0, 1)

#     ms.images(ms.gray2cmap(img2.squeeze()))
#     density = gaussian_filter(ms.t2n(points).astype(float), sigma=1.5)
#     ms.images(ms.gray2cmap(density), win="density")
class GAM_DISC(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set)
        self.discriminator = gans.Discriminator()
        self.feature_extracter = bm.FeatureExtracter()
        self.upsampler = bm.Upsampler(self.feature_extracter.expansion_rate, 1)



    def forward(self, x_input):
        self.eval()
        nn.AdaptiveAvgPool2d
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        mask = self.upsampler.upsample(x_input, x_8s, x_16s, x_32s)

        # Generate Density
        input_size = x_input.shape[-2:]
        points = np.zeros(input_size).ravel()

        n_points = np.random.randint(42, 64)
        points[np.random.choice(points.size, n_points, replace=False)] = 1
        points = points.reshape(input_size)

        # density = gaussian_filter(ms.t2n(points).astype(float), sigma=1.5)
        # density = torch.from_numpy(density).float()

        psf = kernel(points.shape[0], points.shape[1], sigma=10.5)
        density = convolve(points, psf).real; 
        # ms.images(ms.gray2cmap(star_conv.real), win="conv")
        density = torch.from_numpy(density).float()


        return {"mask":mask, "density":density}

    @torch.no_grad()
    def predict(self, batch, **options):
        self.eval()

        n,c,h,w = batch["images"].shape
   

        pred_dict = self(batch["images"].cuda())

        return {"counts":ms.t2n(pred_dict["count"]).ravel()}

    @torch.no_grad()
    def visualize(self, batch, **options):
        pred_dict = self(batch["images"].cuda())
        h, w = batch["images"].shape[-2:]
        mask = ms.t2n(F.interpolate(pred_dict["cam"][None][None], 
                            size=(h,w))).squeeze()
        ms.images(ms.gray2cmap(mask), win="mask")
        
        ms.images(batch["images"], mask>0.5, denorm=1)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'GAS': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, ],
}




