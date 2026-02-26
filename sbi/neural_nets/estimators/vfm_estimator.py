# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Variational Flow Matching Posterior Estimator (VFMPE).

Implements the VFMPE approach from arXiv:2602.13813 ("Pawsterior"), which
predicts interpolation endpoints instead of velocity vectors. Key advantage:
native support for bounded parameter spaces via sigmoid squashing.

Convention (sbi standard): t=0 is data, t=1 is noise.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator
from sbi.utils.vector_field_utils import VectorFieldNet


class VFMEstimator(ConditionalVectorFieldEstimator):
    r"""Variational Flow Matching estimator that predicts interpolation endpoints.

    Instead of directly predicting the velocity v(theta_t, t; x_o), the network
    outputs 2*D values that are split into:
    - mu_data: sigmoid-squashed estimate of theta_0 (data endpoint)
    - mu_noise: unconstrained estimate of theta_1 (noise endpoint)

    The velocity is then: v = mu_noise - mu_data (Eq 23 in the paper).

    The loss is a two-sided endpoint MSE (Eq 21/28):
        L = ||mu_data - theta_0||^2 + ||mu_noise - theta_1||^2

    Time prior: t = 1 - u^{1/(1+alpha)}, u ~ U[0,1]. alpha > 0 emphasizes
    near-data times (t close to 0).
    """

    SCORE_DEFINED: bool = True
    SDE_DEFINED: bool = True
    MARGINALS_DEFINED: bool = True

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        noise_scale: float = 1e-3,
        alpha_time_exponent: float = 0.0,
    ) -> None:
        r"""Creates a VFM estimator.

        Args:
            net: Neural network that outputs 2*D values (endpoints).
            input_shape: Shape of the input theta.
            condition_shape: Shape of the condition x_o.
            embedding_net: Embedding network for the condition.
            noise_scale: sigma_min for numerical stability.
            alpha_time_exponent: Exponent for the time prior.
                alpha=0 gives uniform time sampling. alpha>0 emphasizes
                near-data (t close to 0).
        """
        super().__init__(
            net=net,
            input_shape=input_shape,
            condition_shape=condition_shape,
            embedding_net=embedding_net,
        )
        self.noise_scale = noise_scale
        self.register_buffer(
            "alpha_time_exponent",
            torch.tensor(alpha_time_exponent, dtype=torch.float32),
        )

    def _split_and_activate(self, raw_output: Tensor) -> tuple[Tensor, Tensor]:
        """Split network output into mu_data (sigmoid) and mu_noise (unconstrained).

        Args:
            raw_output: (..., 2*D) network output.

        Returns:
            (mu_data, mu_noise) each of shape (..., D).
        """
        D = raw_output.shape[-1] // 2
        mu_data_raw = raw_output[..., :D]
        mu_noise = raw_output[..., D:]
        mu_data = torch.sigmoid(mu_data_raw)
        return mu_data, mu_noise

    def _sample_time_prior(self, shape: torch.Size, device: torch.device,
                           dtype: torch.dtype) -> Tensor:
        """Sample times from the VFMPE time prior.

        t = 1 - u^{1/(1+alpha)}, u ~ U[0,1].
        When alpha=0, this is uniform. When alpha>0, it emphasizes t near 0.

        Args:
            shape: Shape of the output tensor.
            device: Device for the tensor.
            dtype: Dtype for the tensor.

        Returns:
            Time samples in [0, 1].
        """
        u = torch.rand(shape, device=device, dtype=dtype)
        exponent = 1.0 / (1.0 + self.alpha_time_exponent)
        t = 1.0 - u.pow(exponent)
        return t

    def forward(self, input: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        """Forward pass: returns velocity v = mu_noise - mu_data.

        Args:
            input: theta_t of shape (sample_dim, batch_dim, *event_shape_input).
            condition: x_o of shape (batch_dim, *event_shape_condition).
            time: Time variable in [0,1] of shape (batch_dim,).

        Returns:
            Velocity field v(theta_t, t; x_o).
        """
        batch_shape_input = input.shape[: -len(self.input_shape)]
        batch_shape_cond = condition.shape[: -len(self.condition_shape)]
        batch_shape = torch.broadcast_shapes(
            batch_shape_input, batch_shape_cond,
        )

        condition_emb = self._embedding_net(condition)

        input = torch.broadcast_to(input, batch_shape + self.input_shape)
        condition_emb = torch.broadcast_to(
            condition_emb, batch_shape + condition_emb.shape[len(batch_shape_cond):]
        )
        time = torch.broadcast_to(time, batch_shape)

        input = input.reshape(-1, *input.shape[len(batch_shape):])
        condition_emb = condition_emb.reshape(
            -1, *condition_emb.shape[len(batch_shape):]
        )
        time = time.reshape(-1)

        raw = self.net(input, condition_emb, time)
        mu_data, mu_noise = self._split_and_activate(raw)
        v = mu_noise - mu_data
        v = v.reshape(*batch_shape + self.input_shape)
        return v

    def loss(
        self, input: Tensor, condition: Tensor, times: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Compute the VFMPE two-sided endpoint loss.

        L = ||mu_data - theta_0||^2 + ||mu_noise - theta_1||^2

        Args:
            input: Parameters theta_0 of shape (batch, *input_shape).
            condition: Conditioning variable x_o.
            times: Optional time steps. If None, sampled from time prior.

        Returns:
            Loss of shape (batch,).
        """
        if times is None:
            times = self._sample_time_prior(
                input.shape[:-1], input.device, input.dtype)
        times_ = times[..., None]

        # Sample noise endpoint
        theta_1 = torch.randn_like(input)

        # Interpolation: theta_t = (1-t)*theta_0 + (t+sigma_min)*theta_1
        theta_t = (1 - times_) * input + (times_ + self.noise_scale) * theta_1

        # Get network output at interpolated point
        batch_shape_input = theta_t.shape[: -len(self.input_shape)]
        batch_shape_cond = condition.shape[: -len(self.condition_shape)]
        batch_shape = torch.broadcast_shapes(
            batch_shape_input, batch_shape_cond,
        )

        condition_emb = self._embedding_net(condition)

        theta_t_flat = torch.broadcast_to(
            theta_t, batch_shape + self.input_shape
        ).reshape(-1, *theta_t.shape[len(batch_shape):])
        condition_emb_flat = torch.broadcast_to(
            condition_emb,
            batch_shape + condition_emb.shape[len(batch_shape_cond):],
        ).reshape(-1, *condition_emb.shape[len(batch_shape_cond):])
        times_flat = torch.broadcast_to(times, batch_shape).reshape(-1)

        raw = self.net(theta_t_flat, condition_emb_flat, times_flat)
        raw = raw.reshape(*batch_shape, -1)

        mu_data, mu_noise = self._split_and_activate(raw)

        # Two-sided endpoint MSE
        loss_data = torch.mean((mu_data - input) ** 2, dim=-1)
        loss_noise = torch.mean((mu_noise - theta_1) ** 2, dim=-1)
        return loss_data + loss_noise

    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        r"""ODE function v(theta_t, t; x_o) for sampling.

        Delegates to forward() which returns v = mu_noise - mu_data.
        """
        return self.forward(input, condition, times)

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""Score function derived from the velocity field.

        Same formula as FMPE (same interpolation scheme):
            nabla log p(theta_t | x_o) = (-(1-t)*v - theta_t) / (t + sigma_min)

        Args:
            input: theta_t.
            condition: x_o.
            t: Time.

        Returns:
            Score function.
        """
        v = self(input, condition, t)
        return (-(1 - t) * v - input) / (t + self.noise_scale)

    def drift_fn(
        self, input: Tensor, times: Tensor, effective_t_max: float = 0.99,
    ) -> Tensor:
        r"""Drift function f(t) = -theta_t / (1 - t).

        Same as FMPE (same interpolation scheme).
        """
        return -input / torch.maximum(
            1 - times, torch.tensor(1 - effective_t_max).to(input)
        )

    def diffusion_fn(
        self, input: Tensor, times: Tensor, effective_t_max: float = 0.99,
    ) -> Tensor:
        r"""Diffusion function g(t) = sqrt(2*(t + sigma_min) / (1 - t)).

        Same as FMPE (same interpolation scheme).
        """
        return torch.sqrt(
            2
            * (times + self.noise_scale)
            / torch.maximum(1 - times, torch.tensor(1 - effective_t_max).to(times))
        )

    def mean_t_fn(self, times: Tensor, effective_t_max: float = 0.99) -> Tensor:
        r"""Linear coefficient of the perturbation kernel mean: 1 - t.

        Same as FMPE.
        """
        times = torch.clamp(times, max=effective_t_max)
        mean_t = 1 - times
        for _ in range(len(self.input_shape)):
            mean_t = mean_t.unsqueeze(-1)
        return mean_t

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation of the perturbation kernel: t + sigma_min.

        Same as FMPE.
        """
        std_t = times + self.noise_scale
        for _ in range(len(self.input_shape)):
            std_t = std_t.unsqueeze(-1)
        return std_t
