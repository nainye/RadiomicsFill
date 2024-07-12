import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MET training script.")
    parser.add_argument(
        "--max_pct_mask",
        type=float,
        default=None,
        required=True,
        help="masking rate",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Epoch num",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--GPU",
        type=str,
        default="0",
        help="GPU node num",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="masking rate",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=6,
        help="masking rate",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=1,
        help="masking rate",
    )
    parser.add_argument(
        "--num_head",
        type=int,
        default=1,
        help="transformer head num",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay",
    )

    args = parser.parse_args()

    return args

args = parse_args()

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MET(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 69,
        embedding_dim: int = 64,
        n_head: int = 1,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        dtype = torch.float32
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.dtype = dtype

        # Subtract 1 from desired embedding dim to account for token
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim - 1
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.transformer_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Flatten())
        self.sigmoid = nn.Sigmoid()

        self.mask_embed_layer = nn.Linear(1,1)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        factor = 1.0
    
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.weight.data.normal_(mean=0.0, std=factor * 0.02)
                module.bias.data.zero_()

    def embed_inputs(self, x, idx):
        embd = self.embedding(idx)
        return torch.concat([x.unsqueeze(-1), embd], dim=-1)

    def forward(self, unmasked_x, unmasked_idx, masked_x, masked_idx):
        batch_size = masked_x.size(0)
        fixed_input = torch.ones(batch_size, 1).to(device)
        mask_embed = self.mask_embed_layer(fixed_input)
        seq_len = masked_x.size(1)
        mask_embed = mask_embed.repeat(1, seq_len)
         
        unmasked_inputs = self.embed_inputs(unmasked_x, unmasked_idx)
        masked_inputs = self.embed_inputs(mask_embed, masked_idx)
        
        # Input unmasked_inputs to the encoder
        encoder_output = self.transformer_encoder(unmasked_inputs)
        # Combine encoder output with masked_inputs
        decoder_input = torch.concat([encoder_output, masked_inputs], dim=1)
        # Input decoder_input to the decoder
        decoder_output = self.transformer_decoder(tgt=decoder_input, memory=encoder_output)
        
        x_hat = self.transformer_head(decoder_output)
        x_hat = self.sigmoid(x_hat)

        last_hidden_state = self.final_layer_norm(encoder_output)
        pooled_output = last_hidden_state[:,-1]
        
        return x_hat, pooled_output
    
    def encode(self, x, idx):
        with torch.no_grad():
            inputs = self.embed_inputs(x, idx)
            return self.transformer_encoder(inputs)
        
class METDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs = self.dataset[idx]
        img_id = inputs[0]
        
        tabular_original = inputs[1:].astype(np.float64)

        return torch.tensor(tabular_original), img_id
    
def mask_tensor_1d(x, pct_mask: float = 0.7):
    n = len(x)
    # Adjust to allow masking all but the last 2 clinical variables
    n_maskable = n - 2
    n_masked = int((pct_mask * n_maskable))
    
    # Shuffle indices excluding the last 2
    idx = torch.randperm(n)[:n_maskable]
    # Always set the last 2 indices as unmasked
    always_unmasked_idx = torch.tensor([n-2, n-1], dtype=torch.uint8)
    
    # Calculate masked and unmasked indices
    masked_idx = idx[:n_masked]
    unmasked_idx_temp = idx[n_masked:]
    unmasked_idx = torch.cat((unmasked_idx_temp, always_unmasked_idx), dim=0)  # Add the last 2 indices
    
    # Sort indices
    masked_idx, _ = masked_idx.sort()
    unmasked_idx, _ = unmasked_idx.sort()

    # Initialize tensor to extract unmasked values
    # Initialize unmasked_x with the same dtype as x
    unmasked_x = torch.zeros_like(unmasked_idx, dtype=torch.float)  # Set dtype same as x
    unmasked_x += x[unmasked_idx]

    return unmasked_x, unmasked_idx, masked_idx

def custom_collate_fn(batch, max_pct_mask):
    # Generate pct_mask value to be used across the batch
    # pct_mask = random.uniform(0, max_pct_mask)
    pct_mask = max_pct_mask
    
    # Process batch data
    processed_batch = []
    for data in batch:
        # Apply the same pct_mask to each sample
        tabular_original, img_id = data
        unmasked_x, unmasked_idx, masked_idx = mask_tensor_1d(tabular_original, pct_mask)
        # Add modified data to processed_batch
        processed_batch.append((unmasked_x, unmasked_idx, torch.zeros_like(masked_idx), masked_idx, tabular_original, img_id, pct_mask))
    
    # Combine batch data using torch.utils.data.dataloader.default_collate
    return torch.utils.data.dataloader.default_collate(processed_batch)


def main():
    main_root = "/workspace/data/VinDr-Mammo/"

    trainset = pd.read_csv(os.path.join(main_root, "trainset_normalized_6cls.csv"))
    valset = pd.read_csv(os.path.join(main_root, "valset_normalized_6cls.csv"))

    print("Number of trainset: ", len(trainset))
    print("Number of valset: ", len(valset))

    samples = trainset.iloc[:, 1:]

    # Calculate cosine distance for samples
    distances = cdist(samples, samples, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    epsilon = 1e-5
    rarity_scores = 1 / (min_distances + epsilon)
    weights = rarity_scores / rarity_scores.sum()
    weights /= weights.sum()

    print("Weights: ", weights)

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    train_dataset = METDataset(trainset.values)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda batch: custom_collate_fn(batch, args.max_pct_mask))

    val_dataset = METDataset(valset.values)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, args.max_pct_mask))

    model = MET(num_embeddings=69, embedding_dim=args.embedding_dim, n_head = args.num_head, num_encoder_layers = args.num_encoder_layers,
            num_decoder_layers = args.num_decoder_layers, dim_feedforward = 64, dropout = 0.1).float().to(device)

    if args.pretrained_model_name is not None:
        model.load_state_dict(torch.load(args.pretrained_model_name))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    MSELoss = nn.MSELoss()

    def train_one_epoch(epoch_index):
        running_loss = 0.

        for i, data in enumerate(tqdm(train_loader)):
            unmasked_x, unmasked_idx, masked_x, masked_idx, original, img_id, masking_rate = data
            unmasked_x, masked_x, = unmasked_x.float().to(device), masked_x.float().to(device)
            unmasked_idx, masked_idx = unmasked_idx.to(device), masked_idx.to(device)
            masking_rate = masking_rate.float().to(device)
            original = original.float().to(device)
            
            optimizer.zero_grad(), model.zero_grad()

            recon, pooled_output = model(unmasked_x, unmasked_idx, masked_x, masked_idx)
            recon_loss = MSELoss(original, recon)

            total_loss = recon_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        scheduler.step()  
        avg_loss = running_loss / (i+1)

        return avg_loss

    def validate_one_epoch(epoch_index, data_loader):
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                unmasked_x, unmasked_idx, masked_x, masked_idx, original, img_id, masking_rate = data
                unmasked_x, masked_x, = unmasked_x.float().to(device), masked_x.float().to(device)
                unmasked_idx, masked_idx = unmasked_idx.to(device), masked_idx.to(device)
                masking_rate = masking_rate.float().to(device)
                original = original.float().to(device)
                
                recon, pooled_output = model(unmasked_x, unmasked_idx, masked_x, masked_idx)
                recon_loss = MSELoss(original, recon)

                total_loss = recon_loss

                running_loss += total_loss.item()

        avg_loss = running_loss / (i+1)

        return avg_loss
    
    avg_losses = []
    avg_vlosses = []

    best_vloss = 9999999999999

    best_epoch = -1

    for epoch in range(args.epochs):
        print("** EPOCH {}:".format(epoch))
        now_epochs = list(range(epoch+1))

        model.train(True)

        avg_loss = train_one_epoch(epoch)

        model.eval()

        avg_vloss = validate_one_epoch(epoch, val_loader)

        print('TRAIN loss {}'.format(avg_loss))
        print('VAL loss {}'.format(avg_vloss))

        avg_losses.append(avg_loss)
        avg_vlosses.append(avg_vloss)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), args.model_name.split(".")[0]+"_best_val.pth")
            best_epoch = epoch

        # Start drawing the graph
        plt.figure(figsize=(10,5))

        plt.plot(now_epochs, avg_losses, '-o', label='avg_loss', linewidth=1, markersize=1.5)
        plt.plot(now_epochs, avg_vlosses, '-o', label='avg_vloss', linewidth=1, markersize=1.5)
        plt.title('Average Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)

        # After training, use best_epoch to indicate on the graph
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Validation Epoch')

        # Save the entire graph
        plt.tight_layout()  # Ensure enough space between graphs
        plot_name = args.model_name.split("/")[1].split(".")[0]
        os.makedirs('saved_plot', exist_ok=True)
        plt.savefig('saved_plot/'+plot_name+'.png', dpi=300)
        plt.close('all')


    torch.save(model.state_dict(), args.model_name)

if __name__ == "__main__":
    main()
