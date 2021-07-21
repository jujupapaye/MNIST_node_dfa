import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
import matplotlib.pyplot as plt

DIMENSION = 50

parser = argparse.ArgumentParser('NODE with Direct Feedback Alignment demo')
parser.add_argument('--method', type=str, choices=['DFA', 'adjoint'], default='DFA')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if args.method=='adjoint':
    from torchdiffeq import odeint_adjoint as odeint
    from dfa import odeint_dfa, odeint_dfa2
else:
    from dfa import odeint_dfa, odeint_dfa2
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()
        self.y_t = []

    def forward(self, t, y):
        return torch.mm(y, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                     pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.y_t = []
        self.tab_t = []
        self.net = nn.Sequential(
            nn.Linear(2, DIMENSION),
            nn.Tanh(),
            nn.Linear(DIMENSION, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        with torch.enable_grad():
            output = y.detach()
            output.requires_grad = True
            output = self.net(output)
            self.y_t.append(output)
            self.tab_t.append(t)
        return output




def main():
    nb_expe = 1
    nfe = []
    times = []
    losses = []
    for i in range(nb_expe):
        if args.method == 'DFA':  # DFA backward mode 
            print("Expe ", i)
            ii = 0
            func = ODEFunc().to(device)

            optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
            begin = time.time()
            output_size = 2

            random_matrix = (torch.rand((output_size, output_size)) / np.sqrt(output_size)).to(device)
            #random_matrix = torch.empty(output_size, output_size)
            #torch.nn.init.orthogonal_(random_matrix)
            #random_matrix = (random_matrix / np.sqrt(output_size)).to(device)

            for itr in range(1, args.niters + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch()
                batch_y0, batch_t, batch_y = batch_y0.to(device), batch_t.to(device), batch_y.to(device)
                pred_y = odeint_dfa2(device, func, batch_y0, batch_t)
                for tt in range(len(batch_t)):
                    error = pred_y[tt] - batch_y[tt]
                    nabla = torch.unsqueeze(torch.mm(error.squeeze(), random_matrix), dim=1)
                    nabla.to(device)
                    for j, y_i in enumerate(func.y_t):
                        y_i.backward(nabla, retain_graph=True)
                optimizer.step()
                loss = torch.mean(torch.abs(pred_y - batch_y))

                if itr % args.test_freq == 0:
                    with torch.no_grad():
                        y_plot = odeint_dfa2(device, func, true_y0, t)

                        loss = torch.mean(torch.abs(y_plot - true_y))
                        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                        visualize(true_y, y_plot, func, ii)
                        ii += 1
            end = time.time()
        else:  # normal nn with adjoint method
            ii = 0

            func = ODEFunc().to(device)

            optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
            begin = time.time()

            for itr in range(1, args.niters + 1):
                optimizer.zero_grad()
                batch_y0, batch_t, batch_y = get_batch()
                pred_y = odeint(func, batch_y0, batch_t).to(device)
                #nb_nfe = len(func.y_t)
                #func.y_t = []
                loss = torch.mean(torch.abs(pred_y - batch_y))
                loss.backward()
                optimizer.step()

                if itr % args.test_freq == 0:
                    with torch.no_grad():
                        pred_y = odeint(func, true_y0, t)
                        loss = torch.mean(torch.abs(pred_y - true_y))
                        print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                        visualize(true_y, pred_y, func, ii)
                        ii += 1


        end = time.time()
        #nfe.append(nb_nfe)
        times.append(end-begin)
        if loss.item() < 1:
            losses.append(loss.item())
        print('time:', end-begin)
    print('MEAN TIMES',np.mean(times))
    print('MEAN LOSS',np.mean(losses))
    return times, nfe, losses


if __name__ == '__main__':
    times, nfe, losses = main()
