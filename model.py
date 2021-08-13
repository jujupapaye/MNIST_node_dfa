import torch
import torch.nn as nn
#from torchdiffeq import odeint_adjoint as odeint  # => adjoint method
from torchdiffeq import odeint


MAX_NUM_STEPS = 200

def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y]

class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    """
    Convulutional ODE function.
    """
    def __init__(self, device, img_size, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters
        self.y_t = []

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
        elif non_linearity =='lrelu':
            self.non_linearity = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, t, x):
        self.nfe += 1
        with torch.enable_grad():
            out = x.detach()
            out.requires_grad = True
            if self.time_dependent:
                out = self.conv1(t, x)
                out = self.non_linearity(out)
                out = self.conv2(t, out)
                out = self.non_linearity(out)
                out = self.conv3(t, out)
            else:
                out = self.conv1(x)
                out = self.non_linearity(out)
                out = self.conv2(out)
                out = self.non_linearity(out)
                out = self.conv3(out)
            self.y_t.append(out)
        return out


class ODEBlock(nn.Module):
    """
    ODE Block (ODE func + integration in the forward)
    """
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        del self.odefunc.nfe
        del self.odefunc.y_t
        self.odefunc.nfe = 0
        self.odefunc.y_t = []
        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.odefunc.augment_dim > 0:
            aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
            x = torch.cat([x, aug], 1)


        out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)



class ConvODENet(nn.Module):
    """
    Implements a Convolutional Neural ODE network that contains
    an ODE Block followed by a fully connected linear layer
    to do classification
    """
    def __init__(self, device, img_size, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ConvODENet, self).__init__()
        self.device = device
        self.img_size = img_size  
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ConvODEFunc(device, img_size, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        self.linear_layer = nn.Linear(self.flattened_dim, self.output_dim)

    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        pred = self.linear_layer(features.view(features.size(0), -1))
        if return_features:
            return features, pred
        return pred