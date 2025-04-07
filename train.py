import os
import argparse
import torch
import torch.optim as optim
import random
import os,gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'      #set gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
gc.collect()
torch.cuda.empty_cache()    
from DataGenerator_others import MRIDataGenerator
#from DataGenerator_MRI import MRIDataGenerator          #This is for ADNI data generation

from brainmvp_adapter import BrainMVPDataGenerator
from vit_2d_sa import own_model as create_model
from vit_3d_ga1 import own_model as create_model1
from vit_3d_ga2 import own_model as create_model2
from utils import train_one_epoch, evaluate, test
import dill
warnings.filterwarnings('ignore')        #ignore warnings


def main(args):
    random.seed(3407)  # Set random seed
    
    # Initialize BrainMVP model
    brainmvp_model = None
    if args.use_brainmvp:
        # Load BrainMVP model based on actual implementation
        from brainmvp_model import load_brainmvp_model  # Assuming such function exists
        brainmvp_model = load_brainmvp_model(args.brainmvp_path)
        brainmvp_model = brainmvp_model.to('cuda')
        brainmvp_model.eval()  # Set to evaluation mode
        # Freeze parameters
        for param in brainmvp_model.parameters():
            param.requires_grad = False
    
    # Create enhanced ATOM model
    from enhanced_atom_model import create_enhanced_model
    enhanced_atom = create_enhanced_model(
        brainmvp_model=brainmvp_model,
        brainmvp_embed_dim=args.brainmvp_dim,
        fusion_dim=1000,
        use_brainmvp=args.use_brainmvp
    )
    
    # Data loading
    if args.use_brainmvp:
        # Use BrainMVP-compatible data generator
        with open('./data_pkl/OrganMNIST/train_brainmvp.pkl', 'rb') as f:
            train_dataset = dill.load(f)
        with open('./data_pkl/OrganMNIST/val_brainmvp.pkl', 'rb') as f:
            val_dataset = dill.load(f)
        with open('./data_pkl/OrganMNIST/test_brainmvp.pkl', 'rb') as f:
            test_dataset = dill.load(f)
    else:
        # Use original data generator
        with open('./data_pkl/OrganMNIST/train.pkl', 'rb') as f:
            train_dataset = dill.load(f)
        with open('./data_pkl/OrganMNIST/val.pkl', 'rb') as f:
            val_dataset = dill.load(f)
        with open('./data_pkl/OrganMNIST/test.pkl', 'rb') as f:
            test_dataset = dill.load(f)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)
    
    # Move model to GPU
    enhanced_atom = enhanced_atom.to('cuda')
    
    # Create optimizers
    pg = [p for p in enhanced_atom.vit_2d.parameters() if p.requires_grad] + \
         [q for q in enhanced_atom.vit_3d_ga1.parameters() if q.requires_grad]
    pg1 = [j for j in enhanced_atom.vit_3d_ga2.parameters() if j.requires_grad] + \
          [p for p in enhanced_atom.vit_2d.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-3)
    optimizer2 = optim.AdamW(pg1, lr=args.lr, weight_decay=1e-3)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss, train_loss1, train_high, train_acc, train_acc_k1, x_set = train_one_epoch_enhanced(
            model=enhanced_atom,
            optimizer=optimizer,
            optimizer2=optimizer2,
            data_loader=train_loader,
            device='cuda',
            epoch=epoch
        )
        
        # Validate
        val_loss, val_loss1, val_high, val_acc, val_acc_k1 = evaluate_enhanced(
            model=enhanced_atom,
            data_loader=val_loader,
            device='cuda',
            epoch=epoch
        )
        
        # Test
        test_loss, test_loss1, test_high, test_acc, test_acc_k1 = test_enhanced(
            model=enhanced_atom,
            data_loader=test_loader,
            device='cuda',
            epoch=epoch
        )
        
        # Clean memory
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00005)
    
    # Add BrainMVP-related parameters
    parser.add_argument('--use_brainmvp', action='store_true', default=True,
                        help='whether to use BrainMVP features')
    parser.add_argument('--brainmvp_path', type=str, default='./pretrained/brainmvp.pt',
                        help='path to pre-trained BrainMVP model')
    parser.add_argument('--brainmvp_dim', type=int, default=768,
                        help='dimension of BrainMVP features')
    
    # Original parameters
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--weights2', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--weights3', type=str, default='',
                        help='initial weights path')
    parser.add_argument('data_path', type=str, default='',
                        help='initial data path')

    opt = parser.parse_args()
    main(opt)
