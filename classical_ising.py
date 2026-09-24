import torch
from model import compute_energy


class ClassicalIsingSimulator:
    """
    Standard single-spin flip Metropolis-Hastings MCMC sampler for the Ising Model.
    Serves as the physical benchmark against the GNN global proposal mechanism.
    """
    def __init__(self, J=1.0, h=0.0):
        self.J = J
        self.h = h

    def step(self, spins, edge_index, beta):
        """
        Performs one sweep of single-spin flips across all nodes in random order.
        """
        num_nodes = spins.size(0)
        row, col = edge_index
        spins = spins.clone()
        accepted_flips = 0

        # Random sequence of node visits
        permutation = torch.randperm(num_nodes)

        for node in permutation:
            # Calculate local energy change for flipping spin at 'node'
            neighbors = col[row == node]
            local_field = torch.sum(spins[neighbors])
            
            # Delta E = 2 * s_i * (J * sum_j s_j + h)
            delta_E = 2.0 * spins[node] * (self.J * local_field + self.h)

            # Metropolis acceptance criteria
            if delta_E <= 0 or torch.rand(1).item() < torch.exp(-beta * delta_E).item():
                spins[node] *= -1.0
                accepted_flips += 1

        energy = compute_energy(spins, edge_index, self.J, self.h)
        acceptance_rate = accepted_flips / num_nodes
        return spins, energy, acceptance_rate

    def run_simulation(self, initial_spins, edge_index, num_steps, beta_schedule):
        """
        Runs a full trajectory tracking energy, absolute magnetization, and acceptance rate.
        """
        spins = initial_spins.clone()
        num_nodes = spins.size(0)
        
        energies = []
        magnetizations = []
        acc_rates = []

        for step in range(num_steps):
            beta = beta_schedule[step] if isinstance(beta_schedule, (list, torch.Tensor)) else beta_schedule
            spins, energy, acc_rate = self.step(spins, edge_index, beta)
            
            # Calculate absolute magnetization per spin
            mag = torch.abs(torch.mean(spins)).item()
            
            energies.append(energy.item() / num_nodes)  # Normalize per spin
            magnetizations.append(mag)
            acc_rates.append(acc_rate)

        return spins, energies, magnetizations, acc_rates
