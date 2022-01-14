#!/bin/bash
# file name: test_command.sh

# ./total_process.sh test ../checkpoint/ name model learning rate seed"
# name can be any string while seed has to be an integer

./total_process.sh test ckpt.t7Rep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-10_b256_nGPU1_l256_0 'random_name' 'ResNet18' 0.1 'cifar-10' 7
