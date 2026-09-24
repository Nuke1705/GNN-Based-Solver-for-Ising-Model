import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx


class IsingClusterGNN(nn.Module):
    """
    GNN that predicts bond activation probabilities p_{ij} for adjacent parallel spins.
    Output: p_{ij} in (0.01, 0.99) per edge.
    """
    def __init__(self, in_channels=3, hidden_channels=32):
        super(IsingClusterGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        
        edge_features = torch.cat([h_i, h_j], dim=-1)
        logits = self.edge_mlp(edge_features).squeeze(-1)
        
        # Bond activation probability bounded to avoid log(0)
        p_ij = torch.sigmoid(logits) * 0.98 + 0.01
        return p_ij


def compute_energy(spins, edge_index, J=1.0, h=0.0):
    row, col = edge_index
    interaction_energy = -J * torch.sum(spins[row] * spins[col]) / 2.0
    field_energy = -h * torch.sum(spins)
    return interaction_energy + field_energy


def gnn_cluster_mcmc_step(model, spins, edge_index, beta=1.0, J=1.0, h=0.0):
    num_nodes = spins.size(0)
    row, col = edge_index

    # 1. Feature Prep: [spin, neighbor_sum, beta]
    neighbor_spins = torch.zeros_like(spins)
    neighbor_spins.index_add_(0, row, spins[col])
    beta_tensor = torch.full_like(spins, float(beta))
    x = torch.stack([spins, neighbor_spins, beta_tensor], dim=-1)

    # 2. Forward pass through GNN (retains grad_fn)
    p_ij = model(x, edge_index)

    # 3. Filter aligned edges (s_i == s_j)
    aligned_mask = (spins[row] == spins[col])
    p_baseline = 1.0 - torch.exp(torch.tensor(-2.0 * beta * J))
    p_active = torch.clamp(p_ij * p_baseline, max=0.99) * aligned_mask.float()

    # 4. Sample active bonds (non-differentiable step)
    active_bonds = torch.bernoulli(p_active.detach()).bool()

    # 5. Compute Log-Likelihood log_q of the sampled edge mask (retains grad_fn!)
    active_float = active_bonds.float()
    log_q = torch.sum(
        active_float * torch.log(p_active + 1e-8) +
        (1.0 - active_float) * torch.log(1.0 - p_active + 1e-8)
    )

    # 6. Extract Connected Components via NetworkX
    active_edges = edge_index[:, active_bonds].cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(active_edges.T)
    
    clusters = list(nx.connected_components(G))

    # 7. Randomly select one cluster to flip
    chosen_cluster_idx = torch.randint(0, len(clusters), (1,)).item()
    cluster_nodes = torch.tensor(list(clusters[chosen_cluster_idx]), dtype=torch.long)

    spins_prop = spins.clone()
    spins_prop[cluster_nodes] *= -1.0

    # 8. Metropolis Acceptance Step
    E_curr = compute_energy(spins, edge_index, J, h)
    E_prop = compute_energy(spins_prop, edge_index, J, h)
    delta_E = E_prop - E_curr

    alpha = torch.clamp(torch.exp(-beta * delta_E), max=1.0)

    if torch.rand(1, device=spins.device).item() < alpha.item():
        return spins_prop, E_prop, True, delta_E, log_q
    else:
        return spins, E_curr, False, delta_E, log_q
