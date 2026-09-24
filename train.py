import torch
import torch.optim as optim
import networkx as nx
from model import IsingClusterGNN, gnn_cluster_mcmc_step


def generate_grid_edge_index(grid_size=10):
    G = nx.grid_2d_graph(grid_size, grid_size)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1), grid_size * grid_size


def train_gnn(num_epochs=150, steps_per_epoch=15, grid_size=10, lr=1e-3, save_path="ising_gnn.pt"):
    edge_index, num_nodes = generate_grid_edge_index(grid_size)
    model = IsingClusterGNN(in_channels=3, hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    running_baseline = 0.0

    print(f"Starting Cluster GNN Training ({num_epochs} Epochs)...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        spins = (torch.randint(0, 2, (num_nodes,)) * 2 - 1).float()
        beta = torch.empty(1).uniform_(0.2, 1.2).item()

        epoch_loss = 0.0
        total_accepted = 0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            
            # gnn_cluster_mcmc_step now returns log_q
            spins_next, E_next, accepted, delta_E, log_q = gnn_cluster_mcmc_step(
                model, spins, edge_index, beta=beta
            )

            # Reward formulation: Reward lower system energy
            reward = -E_next.detach().item() / num_nodes
            
            # Advantage with moving average baseline for low variance
            running_baseline = 0.9 * running_baseline + 0.1 * reward
            advantage = reward - running_baseline

            # REINFORCE Policy Gradient Loss: - log_q * advantage
            loss = -log_q * advantage
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if accepted:
                total_accepted += 1
            
            spins = spins_next.detach()

        if epoch % 20 == 0 or epoch == 1:
            acc_rate = (total_accepted / steps_per_epoch) * 100
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {epoch_loss/steps_per_epoch:7.4f} | "
                  f"Acceptance Rate: {acc_rate:5.1f}% | Beta: {beta:.2f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to '{save_path}'.\n")
    return model


if __name__ == "__main__":
    train_gnn()
