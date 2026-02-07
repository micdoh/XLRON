"""Neural network models for reinforcement learning agents."""

# Main actor-critic models
from xlron.models.gnn import (  # Core GNN functions; GNN-based models
    GAT,
    ActorCriticGNN,
    ActorGNN,
    CriticGNN,
    GraphNet,
    GraphNetGAT,
    GraphNetwork,
    add_graphs_tuples,
)
from xlron.models.mlp import (  # Helper functions; MLP-based models
    MLP,
    LaunchPowerActorCriticMLP,
    crelu,
    make_linear_with_orthogonal_init,
    orthogonal_init,
    select_activation,
)
from xlron.models.attention import MultiheadAttention
from xlron.models.transformer import ActorCriticTransformer

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
    "crelu",
    "add_graphs_tuples",
    "orthogonal_init",
    "select_activation",
    "make_linear_with_orthogonal_init",
    # GNN layers
    "GraphNetwork",
    "GraphNetGAT",
    "GAT",
    # Attention layers
    "MultiheadAttention",
    # Transformer models
    "ActorCriticTransformer",
]
