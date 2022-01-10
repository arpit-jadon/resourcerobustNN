python train.py --is-semisup --arch resnet18 --exp-mode pretrain --configs configs/configs.yml\
    --trainer adv --val_method adv --k 1.0  --epochs 20 --save-dense;

<< SKIP
python train.py --arch resnet18 --exp-mode prune --configs configs/configs.yml\
    --trainer adv --val_method adv --k 0.1 --scaled-score-init\
    --source-net pretrained_net_checkpoint_path --epochs 1 --save-dense;

python train.py --arch resnet18 --exp-mode finetune --configs configs/configs.yml\
    --trainer adv --val_method adv --k 0.1 --source-net pruned_net_checkpoint_path\
    --save-dense --lr 0.01 --epochs 1
SKIP
