import torch
for f in ['best_model.pt', 'epoch_20.pt', 'epoch_25.pt', 'final_model.pt']:
    try:
        ckpt = torch.load(f, map_location='cpu')
        print(f"{f:30s} epoch={ckpt.get('epoch',0)+1}  val_loss={ckpt.get('val_loss',99):.4f}  keys={len(ckpt['model_state'])}")
    except: pass