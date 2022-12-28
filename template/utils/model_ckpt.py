import os
import torch



def save_checkpoint(framework, optimizer, curr_epoch, ckpt_dir):
    checkpoint = {
        "net": framework.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": curr_epoch + 1
    }
    checkpoint_path = os.path.join("checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(checkpoint, os.path.join(ckpt_dir, 'ckpt_%s.pth' % (str(curr_epoch))))

def load_checkpoint(args, framework, optimizer, logger):
    logger.info(' 加载断点:{} 恢复训练 '.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)  # 加载断点
    framework.load_state_dict(checkpoint['net'])  # 加载参数
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    args.resume = False
    return start_epoch

def load_model(model, args, logger, device):
    model_path = os.path.join(args.save_dir, "best")
    logger.debug(' 加载模型 {} '.format(model_path))
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.to(device)
    return model

def create_file(path):
    f = open(path, 'w')
    f.close()