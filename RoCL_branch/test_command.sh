#!/bin/bash
# file name: test_command.sh

# ./total_process.sh test ../checkpoint/ name model learning rate seed"
# name can be any string while seed has to be an integer

./total_process.sh test ckpt_300epochs.t7 'random_name' 'ResNet18' 0.1 'cifar-10' 7
