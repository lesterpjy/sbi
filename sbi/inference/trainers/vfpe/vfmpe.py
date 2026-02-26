# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Variational Flow Matching Posterior Estimation (VFMPE).

VFMPE predicts interpolation endpoints instead of velocity vectors, with
native support for bounded parameter spaces via sigmoid squashing.
Based on arXiv:2602.13813.
"""

from typing import Any, Dict, Literal, Optional, Union

from torch.distributions import Distribution
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.posterior_parameters import VectorFieldPosteriorParameters
from sbi.inference.trainers.vfpe.base_vf_inference import VectorFieldTrainer
from sbi.neural_nets.estimators.base import (
    ConditionalEstimatorBuilder,
    ConditionalVectorFieldEstimator,
)
from sbi.neural_nets.factory import posterior_vfm_nn


class VFMPE(VectorFieldTrainer):
    """Variational Flow Matching Posterior Estimation (VFMPE).

    Predicts interpolation endpoints instead of velocity vectors.
    The data endpoint is sigmoid-squashed, providing native support
    for bounded parameter spaces.
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        vf_estimator: Union[
            Literal["mlp"],
            ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator],
        ] = "mlp",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
        **kwargs,
    ) -> None:
        """Initialize VFMPE.

        Args:
            prior: Prior distribution.
            vf_estimator: Neural network architecture or callable builder.
            device: Device to use for training.
            logging_level: Logging level.
            summary_writer: Summary writer for tensorboard.
            show_progress_bars: Whether to show progress bars.
            **kwargs: Additional keyword arguments passed to the default builder.
        """
        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
            vector_field_estimator_builder=vf_estimator,
            **kwargs,
        )

    def build_posterior(
        self,
        vector_field_estimator: Optional[ConditionalVectorFieldEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal["ode", "sde"] = "ode",
        vectorfield_sampling_parameters: Optional[Dict[str, Any]] = None,
        posterior_parameters: Optional[VectorFieldPosteriorParameters] = None,
    ) -> NeuralPosterior:
        r"""Build posterior from the VFM estimator.

        Args:
            vector_field_estimator: The VFM estimator. If None, uses the
                latest trained estimator.
            prior: Prior distribution.
            sample_with: Method for sampling ("ode" or "sde").
            vectorfield_sampling_parameters: Additional ODE/SDE kwargs.
            posterior_parameters: Configuration for VectorFieldPosterior.

        Returns:
            Posterior with .sample() and .log_prob() methods.
        """
        return super().build_posterior(
            estimator=vector_field_estimator,
            prior=prior,
            sample_with=sample_with,
            vectorfield_sampling_parameters=vectorfield_sampling_parameters,
            posterior_parameters=posterior_parameters,
        )

    def _build_default_nn_fn(
        self,
        model: Literal["mlp"],
        **kwargs,
    ) -> ConditionalEstimatorBuilder[ConditionalVectorFieldEstimator]:
        return posterior_vfm_nn(model=model, **kwargs)
