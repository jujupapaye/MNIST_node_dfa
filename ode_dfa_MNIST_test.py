import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--DFA', type=eval, default=True, choices=[True, False])
parser.add_argument('--save_model', type=eval, default=False, choices=[True, False])
parser.add_argument('--name_model', type=str, default='ode_dfa_MNIST.pt')
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=1)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--w_decay', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--print', type=eval, default=True, choices=[True, False])
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args()




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


def main():
    DFA = args.DFA
    img_size = (1, 28, 28)  
    output_dim = 10
    augment_dim = 0
    flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
    

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # get data and define model
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug, args.batch_size, args.test_batch_size)
    model = ConvODENet(device, img_size, num_filters=10,
                       output_dim=output_dim,
                       augment_dim=augment_dim,
                       time_dependent=False,
                       non_linearity='lrelu',
                       adjoint=False)
    #linear_layer = nn.Linear(flattened_dim, output_dim)

    model.to(device)
    #linear_layer.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.w_decay)
    #optimizer2 = torch.optim.Adam(linear_layer.parameters(),lr=args.lr,weight_decay=args.w_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    epoch_loss = 0.
    epoch_nfes = 0
    epoch_backward_nfes = 0
    steps = 0

    loss_func = nn.CrossEntropyLoss().to(device)
    losses=[]
    if DFA:
        #random_mat = torch.randn((output_dim, 28*28)) / np.sqrt(output_dim*28*28)
        random_mat = torch.rand((output_dim, 28*28)) / np.sqrt(output_dim*28*28)
        #random_mat = (-2 * torch.rand((output_dim, 28*28)) + 1 ) / np.sqrt(output_dim*28*28)  # uniform between -1 and 1
        random_mat = random_mat.to(device)
    for epoch in range(args.nepochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            #optimizer2.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            #y_pred = linear_layer(features.view(features.size(0), -1))

            iteration_nfes = model.odeblock.odefunc.nfe
            model.odeblock.odefunc.nfe = 0
            epoch_nfes += iteration_nfes

            if DFA:
                # problem here ?: change the backward
                error = y_pred - to_categorical(y_batch, output_dim).to(device)
                nabla = torch.mm(error, random_mat).to(device)
                optimizer.zero_grad()

                for y_i in model.odeblock.odefunc.y_t:
                    y_i = y_i.view(len(y_batch), -1)
                    y_i.backward(nabla, retain_graph=True)
                loss = loss_func(y_pred, y_batch)
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                loss = loss_func(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())

            for y_i in model.odeblock.odefunc.y_t:
                del y_i.grad
            del loss.grad

            iteration_backward_nfes = model.odeblock.odefunc.nfe
            model.odeblock.odefunc.nfe = 0
            epoch_backward_nfes += iteration_backward_nfes

            if args.print:
                print("\nIteration {}/{}".format(i, len(train_loader)))
                print("Loss: ", loss.item())
            #losses.append(loss.item())

            #print("NFE: {}".format(iteration_nfes))
            #print("BNFE: {}".format(iteration_backward_nfes))
            #print("Total NFE: {}".format(iteration_nfes + iteration_backward_nfes))
            steps += 1

    if args.save_model:
        np.save(args.name_model[:-3]+'losses.npy',losses)
        torch.save(model.state_dict(), args.name_model)
    np.save('losses_dfa.npy', losses)

    total_accuracy = 0
    num = 0
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        total_accuracy += torch.sum(torch.argmax(y_pred,dim=1) == y_batch)
        num += len(y_batch)
    total_accuracy = (total_accuracy / num).item()
    print('Total accuracy on test set:', total_accuracy)



if __name__ == '__main__':
    main()


