import argparse
import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import dgl.nn as dglnn
import tqdm
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear
from dgl.dataloading import DataLoader
from dgl.nn import GraphConv
import hydra
from omegaconf import DictConfig, OmegaConf
from lp_dataset import LinkPredictionDataset
from dgl.nn import GraphConv
from lp_models import GCN, MLP, GATv2, SAGE

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")
log = logging.getLogger(__name__)

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g = g.to("cpu")
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    g_simple = g_simple.to(g.device)
    reverse_mapping = reverse_mapping.to(g.device)
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

class Logger:
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            log.info(f'Run {run + 1:02d}:')
            log.info(f'Highest Valid: {result[:, 0].max():.2f}')
            log.info(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            log.info(f'All runs:')
            r = best_result[:, 0]
            log.info(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            log.info(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

def train_link_prediction(cfg, device, dataset, model, run):
    train_data = dataset.get_edge_split()['train']
    val_data = dataset.get_edge_split()['val']
    train_edges, train_labels = train_data['edges'], train_data['labels']
    val_edges, val_labels = val_data['edges'], val_data['labels']

    train_dataloader = torch.utils.data.DataLoader(
        dataset=list(zip(train_edges, train_labels)),
        batch_size=cfg.batch_size,
        shuffle=True
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val_acc = 0
    best_epoch = 0
    early_stop_count = 0

    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    checkpoint_path = os.path.join(
        cfg.checkpoint_folder, f"{cfg.model_name}_{cfg.dataset}_best.pth"
    )

    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            edges, labels = batch
            edges, labels = edges.to(device), labels.to(device)

            src_features = dataset.graph.ndata['feat'][edges[:, 0]]
            dst_features = dataset.graph.ndata['feat'][edges[:, 1]]

            preds = model(dataset.graph, edges[:, 0], edges[:, 1]).squeeze()

            loss = F.binary_cross_entropy(preds, labels.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        log.info(f"Run {run}, Epoch {epoch + 1}/{cfg.n_epochs}, Loss: {avg_loss:.4f}")

        val_acc = evaluate_link_prediction(dataset, model, 'val', cfg.model_name)
        log.info(f"Run {run}, Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"Run {run}: Best model saved at epoch {best_epoch} with Validation Accuracy: {best_val_acc:.4f}")
        else:
            early_stop_count += 1

        if early_stop_count >= cfg.early_stop_patience:
            log.info(f"Run {run}: Early stopping triggered at epoch {epoch + 1}")
            break

    log.info(f"Run {run}: Training finished. Best Epoch: {best_epoch}, Best Validation Accuracy: {best_val_acc:.4f}")

def evaluate_link_prediction(dataset, model, split, model_name):
    data = dataset.get_edge_split()[split]
    edges, labels = data['edges'], data['labels']
    model.eval()

    with torch.no_grad():
        preds = model(dataset.graph, edges[:, 0], edges[:, 1]).squeeze()

        preds = (preds > 0.5).float()
        acc = (preds == labels.float()).sum().item() / len(labels)
    return acc

@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base="1.2")
def main(cfg):
    if not os.path.isfile("config.yaml"):
        OmegaConf.save(config=cfg, f=os.path.join("config.yaml"))

    log.info("Loading data")
    data_path = '/scratch/ys6310/Mario_All_Embs/mm-graph/'
    dataset_name = cfg.dataset
    feature_path = data_path + dataset_name
    feat_name = cfg.feat
    device = torch.device('cpu' if cfg.mode == 'cpu' else 'cuda')
    dataset = LinkPredictionDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        neg_sampling_ratio=1.0,
        device=device
    )

    g = dataset.graph
    g = dgl.remove_self_loop(g)
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to(device)

    if cfg.model_name == "SAGE":
        model = SAGE(g.ndata['feat'].shape[1], cfg.hidden_dim, 1, cfg.num_layers).to(device)
    elif cfg.model_name == "GCN":
        model = GCN(g.ndata['feat'].shape[1], cfg.hidden_dim, 1, cfg.num_layers).to(device)
    elif cfg.model_name == "MLP":
        model = MLP(g.ndata['feat'].shape[1], cfg.hidden_dim, 1, cfg.num_layers).to(device) 
    elif cfg.model_name == "GATv2":
        model = GATv2(g.ndata['feat'].shape[1], cfg.hidden_dim, cfg.num_layers, cfg.heads, 1).to(device)

    logger = Logger(cfg.runs)

    for run in range(cfg.runs):
        log.info(f"Starting run {run + 1}/{cfg.runs}")
        train_link_prediction(cfg, device, dataset, model, run)
        val_acc = evaluate_link_prediction(dataset, model, 'val', cfg.model_name)
        test_acc = evaluate_link_prediction(dataset, model, 'test', cfg.model_name)
        log.info(f"Run {run + 1}: Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        logger.add_result(run, (val_acc, test_acc))

    logger.print_statistics()

if __name__ == "__main__":
    main()
