"""Neural network models for reinforcement learning agents."""

# Main actor-critic models
from xlron.models.models import (
    # MLP-based models
    ActorCriticMLP,
    LaunchPowerActorCriticMLP,
    MLP,
    # GNN-based models
    ActorCriticGNN,
    ActorGNN,
    CriticGNN,
    GraphNet,
    # Helper functions
    make_dense_layers,
    crelu,
    add_graphs_tuples,
)

# Graph neural network layers
from xlron.models.gnn import (
    # Core GNN functions
    GraphNetwork,
    GraphNetGAT,
    GAT,
    GraphConvolution,
    GraphMapFeatures,
    # Other GNN architectures
    InteractionNetwork,
    RelationNetwork,
    DeepSets,
    # Helper functions
    add_self_edges_fn,
)

__all__ = [
    # MLP models
    "ActorCriticMLP",
    "LaunchPowerActorCriticMLP",
    "MLP",
    # GNN models
    "ActorCriticGNN",
    "ActorGNN",
    "CriticGNN",
    "GraphNet",
    # Model helpers
    "make_dense_layers",
    "crelu",
    "add_graphs_tuples",
    # GNN layers
    "GraphNetwork",
    "GraphNetGAT",
    "GAT",
    "GraphConvolution",
    "GraphMapFeatures",
    "InteractionNetwork",
    "RelationNetwork",
    "DeepSets",
    "add_self_edges_fn",
]