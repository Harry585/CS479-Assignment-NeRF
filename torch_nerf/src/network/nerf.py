"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim
        layer_sizes = [pos_dim, feat_dim, feat_dim, feat_dim, feat_dim, feat_dim + pos_dim,  \
                       feat_dim, feat_dim, feat_dim, ]
        self.MLP = nn.ModuleList([
            nn.Linear(pos_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim + pos_dim),
            nn.Linear(feat_dim + pos_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        ])
        self.predict_sigma = nn.Linear(feat_dim, 1)
        self.no_activation_layer = nn.Linear(feat_dim, feat_dim)
        self.predict_rgb = nn.Sequential(
                nn.Linear(feat_dim + view_dir_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Sigmoid()
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """
        # NOTE: we use a skip connection, so we must splice in the input
        h = pos
        for i, layer in enumerate(self.MLP[:4]):
            h = layer(h)
            print(f"layer {i+1}: h shape={h.shape}")
            h = F.relu(h)
        # Splice the position input in again at layer 5
        h = torch.cat([h, pos], dim=-1)
        for i, layer in enumerate(self.MLP[4:7]):
            h = layer(h)
            print(f"layer {i+1}: h shape={h.shape}")
            h = F.relu(h)

        # Get sigma
        sigma = F.relu(self.predict_sigma(h))

        # Get rgb values
        h = self.no_activation_layer(h)
        h = torch.cat([h, view_dir], dim=-1)
        rgb = self.predict_rgb(h)
        return sigma, rgb



