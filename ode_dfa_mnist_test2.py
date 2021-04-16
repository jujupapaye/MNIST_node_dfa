import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-2)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--w_decay', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

MAX_NUM_STEPS = 500


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
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        self.odefunc.nfe = 0
        self.odefunc.t_eval = []
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


def get_mnist_loaders(data_aug=False, batch_size=64, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=False, transform=transform_train),
        batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

if __name__ == '__main__':

    img_size = (1, 28, 28)
    output_dim = 10
    augment_dim = 0

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    model = ConvODENet(device, img_size, num_filters=10,
                       output_dim=output_dim,
                       augment_dim=augment_dim,
                       time_dependent=False,
                       non_linearity='lrelu',
                       adjoint=False)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.w_decay)

    epoch_loss = 0.
    epoch_nfes = 0
    epoch_backward_nfes = 0
    print_freq = 1
    record_freq = 10
    steps = 0
    #histories = {'loss_history': [], 'nfe_history': [],
                      #'bnfe_history': [], 'total_nfe_history': [],
                      #'epoch_loss_history': [], 'epoch_nfe_history': [],
                      #'epoch_bnfe_history': [], 'epoch_total_nfe_history': []}
    #buffer = {'loss': [], 'nfe': [], 'bnfe': [], 'total_nfe': []}
    loss_func = nn.CrossEntropyLoss(reduction='none') #nn.MSELoss() # nn.CrossEntropyLoss()
    losses=[]
    #torch.manual_seed(0)
    random_mat = torch.randn((output_dim, 28*28)) / np.sqrt(output_dim*28*28)
    print(random_mat[0][0])
    random_mat.to(device)
    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        #print("y_pred", y_pred.size())

        iteration_nfes = model.odeblock.odefunc.nfe
        model.odeblock.odefunc.nfe = 0
        epoch_nfes += iteration_nfes

        error = y_pred - to_categorical(y_batch, output_dim)
        nabla = torch.mm(error, random_mat)
        nabla.to(device)
        for y_i in model.odeblock.odefunc.y_t:
            y_i = y_i.view(len(y_batch), -1)
            y_i.backward(nabla, retain_graph=True)

        loss = loss_func(y_pred, y_batch)
        loss = loss.sum()
        loss.backward(retain_graph=True)

        optimizer.step()

        iteration_backward_nfes = model.odeblock.odefunc.nfe
        model.odeblock.odefunc.nfe = 0
        epoch_backward_nfes += iteration_backward_nfes

        print("\nIteration {}/{}".format(i, len(train_loader)))
        print("Loss: ", loss)
        losses.append(loss)

        print("NFE: {}".format(iteration_nfes))
        print("BNFE: {}".format(iteration_backward_nfes))
        print("Total NFE: {}".format(iteration_nfes + iteration_backward_nfes))
        steps += 1


