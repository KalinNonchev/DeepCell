from matplotlib import pyplot as plt
from torch.backends import cudnn
import numpy as np
import random
import torch
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import anndata as ad
import lightning as L
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def model_fine_tune(model, dataloader, rho=False, gene_expression=True, max_epochs=10):
    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    if rho:
        # Unfreeze the 'gene_expression' layer
        for param in model.rho.parameters():
            param.requires_grad = True
        # Unfreeze the 'gene_expression' layer
    if gene_expression:
        for param in model.gene_expression.parameters():
            param.requires_grad = True
    
    # Step 2: Update the optimizer to include only the trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=model.hparams.lr, weight_decay=model.hparams.weight_decay
    )
    model.eval()

    # Set 'rho' and 'gene_expression' layers to train mode (since they are fine-tuned)
    if rho:
        model.rho.train()
    if gene_expression:
        model.gene_expression.train()
    # Step 3: Train the model using the fine-tuning DataLoader
    trainer = L.Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

def run_inference_from_dataloader(model, dataloader, device, predict_genes=True):
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            if type(X) is list:
                X = (x.to(device) for x in X)
            else:
                X = X.to(device)
            y = model.forward(X, predict_genes=predict_genes)
            #y_zeros = y_zeros.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.array(out)

def get_balanced_index(barcode, labels, n_count):
    labels = np.array(labels)
    resampled_barcodes = []

    for label in np.unique(labels):
        resampled_barcodes.extend(np.random.choice(barcode[labels == label], size=n_count - 1))
    return resampled_barcodes


def run_default(adata, resolution=1.0):
    adata = adata.copy()
    n_comps = min(adata.shape[1] - 1, 50)
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    #sc.tl.umap(adata)
    #sc.pl.umap(adata, color='leiden')
    adata.obsm["latent"] = adata.obsm["X_pca"]
    adata.obs["label"] = adata.obs["leiden"].values
    return adata


def run_aestetik(adata, window_size=3, resolution=1.0):
    from aestetik import AESTETIK
    
    adata = adata.copy()
    sc.pp.pca(adata)
    
    # we set the transcriptomics modality
    adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"]

    # we set the morphology modality
    adata.obsm["X_pca_morphology"] = np.ones((len(adata), 5)) # dummy number to keep dim low
    
    resolution = float(resolution) # leiden with resolution
    model = AESTETIK(adata,
                 clustering_method="leiden",
                 nCluster=resolution,
                 window_size=window_size, 
                     morphology_weight=0)

    model.prepare_input_for_model()
    model.train()
    model.compute_spot_representations(cluster=True, n_repeats=10)
    adata = model.adata.copy()
    adata.obsm["latent"] = adata.obsm["AESTETIK"]
    adata.obs["label"] = adata.obs["AESTETIK_cluster"].values
    return adata
   


def spatial_upsample_and_smooth(counts, obs, barcode, resolution, smooth_n=0, augmentation="default", standardize_per_sample=True):
    samples = np.array([b.split('_')[1] for b in barcode])
    unqiue_samples = np.unique(samples)
    resampled_barcodes = []
    transcriptomics_smooth = np.zeros(counts.shape)
    for sample in tqdm(unqiue_samples):
        idx = samples == sample
        sample_barcode = barcode[idx]
        adata = ad.AnnData(counts[idx, :], obs=obs.iloc[idx])

        if resolution:
            if augmentation == "aestetik":
                adata = run_aestetik(adata, resolution=resolution)
            elif augmentation == "default": 
                adata = run_default(adata, resolution=resolution)
            else:
                # If none of the above conditions are met, raise a NotImplementedError
                raise NotImplementedError(f"Not implemented: {augmentation}")
        
            #sc.pl.umap(adata, color='leiden')
            #most_common, num_most_common = Counter(adata.obs.label).most_common(1)[0]
            n_count = np.max(adata.obs.label.value_counts()).astype(int)
            resampled_barcodes.extend(get_balanced_index(sample_barcode, adata.obs.leiden, n_count))#num_most_common
        else:
            resampled_barcodes.extend(sample_barcode)

        if standardize_per_sample:
            scaler = StandardScaler()
            transcriptomics_smooth[idx,:] = scaler.fit_transform(counts[idx, :])

        if smooth_n > 0:
            neigh = KNeighborsRegressor(n_neighbors=smooth_n, n_jobs=-1)
            neigh.fit(adata.obsm['latent'], adata.X)
            transcriptomics_smooth[idx,:] = neigh.predict(adata.obsm['latent'])
            
        

    return np.array(resampled_barcodes), transcriptomics_smooth


def add_zero_padding(original_array, desired_padding):
    if desired_padding == original_array.shape[0]:
        return original_array
    # Calculate the amount of padding needed
    padding_needed = desired_padding - original_array.shape[0]
    
    padded_array = np.pad(original_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)

    return padded_array
    
def load_multiple_pickles(files):
    return pd.concat([pd.read_pickle(f) for f in files])

def log1p_normalization(arr, factor=1e2):
    # max_vals = arr.max(axis=1, keepdims=True)
    return np.log1p((arr.T/np.sum(arr, axis=1)).T * factor)

def load_data(samples, out_folder, feature_model=None, cell_radius=None, load_image_features=True, factor=1e2, raw_counts=False, min_count_cell=0):
    barcode_list = []
    image_features_emb = []
    gene_expr = []

    expr_files = [f"{out_folder}/data/expression/{sample}.pkl" for sample in samples]

    gene_expr = load_multiple_pickles(expr_files)
    
    barcode_list = gene_expr.index.values
    gene_expr = gene_expr.values
    
    if load_image_features:
        if isinstance(cell_radius, int) or isinstance(cell_radius, str):
            cell_radius = [cell_radius]
        if isinstance(feature_model, str):
            feature_model = [feature_model]
        image_features_emb = []
        for model in feature_model:
            for radius in cell_radius:
                image_features_files = [f"{out_folder}/data/image_features/{sample}_{model}_{radius}.pkl" for sample in samples]
                emb = load_multiple_pickles(image_features_files)
                emb = emb.loc[barcode_list]
                image_features_emb.append(emb.values)
        
        image_features_emb = np.concatenate(image_features_emb, axis=1)

    data = {}
    
    if not raw_counts:
        if min_count_cell:
            count_per_cell = gene_expr.sum(axis=1)
            keep_cells = count_per_cell > min_count_cell
            gene_expr = gene_expr[keep_cells]
            barcode_list = barcode_list[keep_cells]
            if load_image_features:
                image_features_emb = image_features_emb[keep_cells]
                
        gene_expr = log1p_normalization(gene_expr)
        
    data["y"] = gene_expr
    
    if load_image_features:
        data["X"] = image_features_emb
        
    data["barcode"] = barcode_list
    return data

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def plot_loss_values(train_losses, val_losses=None):
    train_losses = np.array(train_losses)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    train_idx = np.arange(0, len(train_losses))
    plt.plot(train_idx, train_losses, color="b", label="train")

    if val_losses is not None:
        val_losses = np.array(val_losses)
        val_idx = np.arange(0, len(val_losses)) * (len(train_losses) // len(val_losses) + 1)
        plt.plot(val_idx, val_losses, color="r", label="val")

    plt.legend()