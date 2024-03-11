from dataclasses import dataclass, field
from enum import Enum


@dataclass
class GraphTransformerConfig:
    num_heads: int = 2
    ln_type: str = "pre"
    num_mlp_layers: int = 0


class SeqPosEnc(int, Enum):
    Pos = 0
    Rotary = 1


@dataclass
class SeqTransformerConfig:
    num_heads: int = 2
    posenc: SeqPosEnc = SeqPosEnc.Rotary


@dataclass
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
    seq_transformer: SeqTransformerConfig = field(default_factory=SeqTransformerConfig)
