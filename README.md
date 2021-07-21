# Using Direct Feedback Alignment for training Neural Ordinary Differential Equations

Requirements : torchdiffeq. (pip install torchdiffeq)

2 fichiers éxécutables pour 2 expériences + pdf explication et résulats.

## Spiral dataset test

usage: python ode_dfa_spiral_test.py [-h] [--method {DFA,adjoint}]
                                                [--data_size DATA_SIZE]
                                                [--batch_time BATCH_TIME]
                                                [--batch_size BATCH_SIZE]
                                                [--niters NITERS]
                                                [--test_freq TEST_FREQ]
                                                [--viz] [--gpu GPU]


## MNIST classification test

usage : python ode_dfa_MNIST_test.py [-h] [--DFA {True,False}]
                             [--name_model NAME_MODEL]
                             [--adjoint {True,False}] [--nepochs NEPOCHS]
                             [--data_aug {True,False}] [--lr LR]
                             [--w_decay W_DECAY] [--batch_size BATCH_SIZE]
                             [--test_batch_size TEST_BATCH_SIZE] [--gpu GPU]
                              
where :
- DFA : if True, we use DFA to do the back propagation in the training, default=True
- save_model : if True we save the model that we learned, default=True
- NAME_MODEL : the name of the model that we learned (if we save it),default='ode_dfa_MNIST.pt'
- adjoint : if True ,default=False
- nepochs : number of epochs in the training, default=1
- data_aug : if True we do data augmentation,default=True
- lr : learning rate,default=0.001
- w_decay : weight decay, default=0.1
- batch_size: size of a batch for the training, default=128
- test_batch_size : size of a batch for the test,default=1000
- gpu : number of the gpu that we will use, if gpu is not available we use cpu,default=1

