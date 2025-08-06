
from config.config import *

def old_run_data():
    # from local
    checkpoint_path = os.path.join(old_checkpoint_dir, f"checkpoint_epoch_{old_run_checkpoint_no}.pth")
    
    checkpoint = torch.load(checkpoint_path)

    start_epoch = checkpoint['epoch']
    print(f"✅ Resuming from epoch {start_epoch} with loss: {checkpoint['loss']}")

    return checkpoint, start_epoch
