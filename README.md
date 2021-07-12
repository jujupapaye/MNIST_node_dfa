# Using Direct Feedback Alignment for training NeuralOrdinary Differential Equations

Requirements : torchdiffeq. (pip install torchdiffeq)

2 fichiers éxécutables pour 2 expériences + pdf explication et résulats.

## Spiral dataset test

usage : python ode_dfa_spiral_test.py [-h] [--method {dopri5,adams}] [--data_size DATA_SIZE]
                [--batch_time BATCH_TIME] [--batch_size BATCH_SIZE]
                [--niters NITERS] [--test_freq TEST_FREQ] [--viz] [--gpu GPU]
                [--adjoint]


## MNIST classification test

usage : python ode_dfa_MNIST_test.py [-h] [--DFA {True,False}]
                             [--name_model NAME_MODEL]
                             [--adjoint {True,False}] [--nepochs NEPOCHS]
                             [--data_aug {True,False}] [--lr LR]
                             [--w_decay W_DECAY] [--batch_size BATCH_SIZE]
                             [--test_batch_size TEST_BATCH_SIZE] [--gpu GPU]
                              
where :
- NAME_MODEL : ,default='.pt'
- adjoint : ,default=False
- nepochs : ,default=1
- data_aug : ,default=True
- lr : ,default=0.001
- w_decay : default=0.1
- batch_size: default=128
- test_batch_size : ,default=1000
- gpu : ,default=1

