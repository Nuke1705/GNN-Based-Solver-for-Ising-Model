import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from model import IsingClusterGNN, gnn_cluster_mcmc_step, compute_energy
from classical_ising import ClassicalIsingSimulator
from train import train_gnn, generate_grid_edge_index


def run_benchmark():
    # --- 1. Configuration & Setup ---
    grid_size = 10                  # 10x10 lattice = 100 spins
    num_steps = 500                 # Number of relaxation steps
    weights_path = "ising_gnn.pt"
    
    edge_index, num_nodes = generate_grid_edge_index(grid_size)

    # --- 2. Load or Train GNN Model ---
    model = IsingClusterGNN(in_channels=3, hidden_channels=32)
    if not os.path.exists(weights_path):
        print("Trained model weights not found. Training model now...")
        model = train_gnn(num_epochs=150, save_path=weights_path)
    else:
        print(f"Loading pre-trained model weights from '{weights_path}'...")
        model.load_state_dict(torch.load(weights_path))
    model.eval()

    # --- 3. Define Temperature Annealing Schedule ---
    # Cool system linearly from high temperature (beta=0.2) to low temperature (beta=1.5)
    beta_schedule = torch.linspace(0.2, 1.5, num_steps)

    # --- 4. Initialize Identical Initial Random State ---
    torch.manual_seed(42)
    initial_spins = (torch.randint(0, 2, (num_nodes,)) * 2 - 1).float()

    print("\n--- Running Relaxation Benchmark ---")

    # --- 5. Run Classical Metropolis Simulator ---
    classical_sim = ClassicalIsingSimulator(J=1.0, h=0.0)
    cspins, class_energies, class_mags, class_accs = classical_sim.run_simulation(
        initial_spins, edge_index, num_steps, 0.5
    )

    # --- 6. Run GNN-MCMC Simulator ---
    gnn_spins = initial_spins.clone()
    gnn_energies = []
    gnn_mags = []
    gnn_accs = []
    
    accepted_counter = 0

    for step in range(num_steps):
        ##beta = beta_schedule[step].item()
        beta = 0.5

        gnn_spins, E_curr, accepted, delta_E, log_q = gnn_cluster_mcmc_step(
            model, gnn_spins, edge_index, beta=beta
        )
        if accepted:
            accepted_counter += 1

        mag = torch.abs(torch.mean(gnn_spins)).item()
        gnn_energies.append(E_curr.item() / num_nodes)
        gnn_mags.append(mag)
        gnn_accs.append(accepted_counter / (step + 1))

    # --- 7. Plot Comparison Graphs ---
    steps = np.arange(1, num_steps + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Energy per Spin vs Steps
    axes[0].plot(steps, class_energies, label="Classical Metropolis", color="crimson", alpha=0.8)
    axes[0].plot(steps, gnn_energies, label="GNN-MCMC (Proposed)", color="navy", alpha=0.8)
    axes[0].set_title("Energy Relaxation ($E/N$)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("MCMC Step")
    axes[0].set_ylabel("Energy per Spin")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Plot 2: Absolute Magnetization vs Steps
    axes[1].plot(steps, class_mags, label="Classical Metropolis", color="crimson", alpha=0.8)
    axes[1].plot(steps, gnn_mags, label="GNN-MCMC (Proposed)", color="navy", alpha=0.8)
    axes[1].set_title("Magnetization Decay ($|M|/N$)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("MCMC Step")
    axes[1].set_ylabel("Absolute Magnetization")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Plot 3: Acceptance Rate vs Steps
    axes[2].plot(steps, class_accs, label="Classical Sweep Acceptance", color="crimson", alpha=0.8)
    axes[2].plot(steps, gnn_accs, label="GNN Proposal Acceptance", color="navy", alpha=0.8)
    axes[2].set_title("Cumulative Acceptance Rate", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("MCMC Step")
    axes[2].set_ylabel("Acceptance Probability")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("ising_gnn_vs_classical.png", dpi=300)
    print("Benchmark complete! Plot saved to 'ising_gnn_vs_classical.png'.")
    plt.show()

    print(gnn_spins)
    print(cspins)
    print(initial_spins)

if __name__ == "__main__":
    run_benchmark()
