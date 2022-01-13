from models.resnet import test
def get_model(args):

    if args.dataset == 'cifar-10':
        num_classes=10
    elif args.dataset == 'cifar-100':
        num_classes=100
    else:
        raise NotImplementedError

    if 'contrastive' in args.train_type or 'linear_eval' in args.train_type:
        contrastive_learning=False  
    else:
        contrastive_learning=False

    if args.model == 'ResNet18':
        model = test(contrastive_learning)
        print('ResNet18 is loading ...')
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes,contrastive_learning)
        print('ResNet 50 is loading ...')
    return model
