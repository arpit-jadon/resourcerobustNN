<< SKIP
python train.py --arch resnet18 --exp-mode pretrain --configs configs/configs.yml\
    --trainer base --val_method adv --k 1.0  --epochs 20 --save-dense --exp-name res18_base_adv

python train.py --arch resnet18 --exp-mode prune --configs configs/configs.yml\
    --trainer base --val_method adv --k 0.1 --scaled-score-init --exp-name res18_base_adv\
    --source-net ./trained_models/res18_base_adv/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs 20 --save-dense

python train.py --arch resnet18 --exp-mode finetune --configs configs/configs.yml\
    --trainer base --val_method adv --k 0.1 --source-net ./trained_models/res18_base_adv/prune/latest_exp/checkpoint/checkpoint.pth.tar\
    --save-dense --lr 0.01 --epochs 20 --exp-name res18_base_adv
SKIP

<< SKIP
SKIP

python train.py --arch resnet18 --exp-mode pretrain --configs configs/configs.yml\
    --trainer adv --val_method adv --k 1.0  --epochs 25 --save-dense --exp-name rocl_ext_adv_base\
    --source-net ./trained_models/rocl_ckpt_same_attack --load_RoCL extractor

python train.py --arch resnet18 --exp-mode prune --configs configs/configs.yml\
    --trainer base --val_method adv --k 0.1 --scaled-score-init --exp-name rocl_ext_adv_base\
    --source-net ./trained_models/rocl_ext_adv_base/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs 25 --save-dense

python train.py --arch resnet18 --exp-mode finetune --configs configs/configs.yml\
    --trainer base --val_method adv --k 0.1 --source-net ./trained_models/rocl_ext_adv_base/prune/latest_exp/checkpoint/checkpoint.pth.tar\
    --save-dense --lr 0.01 --epochs 25 --exp-name rocl_ext_adv_base

<< skip
python train.py --arch resnet18 --exp-mode pretrain --configs configs/configs.yml\
    --epochs 1 --save-dense --exp-name res18_base_adv --black_box_eval
skip