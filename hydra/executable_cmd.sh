<< SKIP
python train.py --arch resnet18 --exp-mode pretrain --configs configs/configs.yml\
    --trainer adv --val_method adv --k 1.0  --epochs 2 --save-dense
SKIP


python train.py --arch resnet18 --exp-mode prune --configs configs/configs.yml\
    --trainer adv --val_method adv --k 0.1 --scaled-score-init\
    --source-net ./trained_models/temp/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs 1 --save-dense;

<< SKIP
python train.py --arch resnet18 --exp-mode finetune --configs configs/configs.yml\
    --trainer adv --val_method adv --k 0.1 --source-net pruned_net_checkpoint_path\
    --save-dense --lr 0.01 --epochs 1
SKIP
