# import logging, time, os, warnings, signal
import numpy as np
from scipy.special import factorial as sp_factorial, erf
# from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# from scipy.ndimage import gaussian_filter1d as gauss_filter
# from scipy.interpolate import interp1d
import itertools

from .HierarchicalBayesModel import HierarchicalModel
from .HierarchicalBayesModel.structures import (
    build_distr_structure_from_params,
    build_key,
    prior_structure,
    parse_name_and_indices,
)
from .HierarchicalBayesModel.functions import (
    halfnorm_ppf,
    norm_ppf,
    bounded_flat,
    norm_cdf,
)

from placefield_dynamics.placefield_detection.utils import (
    gauss_smooth,
    get_spiking_data,
)

from .utils import gabor_filter, sine_grating, softplus

# from .utils import circmean as weighted_circmean, model_of_tuning_curve
# from .analyze_results import build_inference_results


# logging.basicConfig(level=logging.ERROR)

# warnings.filterwarnings("ignore")

# logger = logging.getLogger("ultranest")
# logger.addHandler(logging.NullHandler())
# logger.setLevel(logging.WARNING)


class HierarchicalBayesInference(HierarchicalModel):

    def prepare_data(self, event_counts, T, measure_points, FoV_range, FoV_steps=51):

        super().prepare_data(
            event_counts, T, ["phases", "angles", "cycles"], [False, False, False]
        )

        self.measure_points = measure_points
        self.set_FoV(FoV_range, FoV_steps)
        self.set_gratings(FoV_steps)

        ## pre-calculate log-factorial for speedup
        self.log_N_factorial = np.log(sp_factorial(self.data["event_counts"]))

        # self.log = logging.getLogger("nestLogger")
        # self.log.setLevel(logLevel)

    def set_FoV(self, FoV_range, FoV_steps):
        print("set FoV")

        self.X_FoV, self.Y_FoV = np.meshgrid(
            np.linspace(-FoV_range[0] / 2, FoV_range[0] / 2, FoV_steps),
            np.linspace(-FoV_range[1] / 2, FoV_range[1] / 2, FoV_steps),
        )
        self.FoV_grid = np.dstack((self.X_FoV, self.Y_FoV))
        # self.X_FoV = self.X_FoV[np.newaxis, ...]
        # self.Y_FoV = self.Y_FoV[np.newaxis, ...]

    def set_gratings(self, FoV_steps):
        print("set gratings")

        self.gratings = np.zeros(
            (
                self.dims["n_phases"],
                self.dims["n_angles"],
                self.dims["n_cycles"],
                FoV_steps,
                FoV_steps,
            )
        )
        for prod in itertools.product(
            enumerate(self.measure_points[0]),
            enumerate(self.measure_points[1]),
            enumerate(self.measure_points[2]),
        ):
            idx, elems = zip(*prod)
            phi_0, theta, f = elems
            # print(elems)
            self.gratings[*idx, ...] = sine_grating(
                self.X_FoV,  # [0, ...],
                self.Y_FoV,  # [0, ...],
                theta,
                f,
                phi_0,
                square=True,
            )

    def set_priors(self, priors_init=None):

        if priors_init is None:

            fmap = self.data["event_counts"] / self.data["T"]
            A0_guess, A_guess = np.maximum(np.percentile(fmap, [20, 80]), 0.1)

            self.priors_init = {}

            self.priors_init["nl_offset"] = prior_structure(
                halfnorm_ppf,
                loc=1.0,
                scale=A0_guess,
            )
            self.priors_init["nl_amplitude"] = prior_structure(
                halfnorm_ppf,
                loc=1.0,
                scale=5.0,
            )
            # self.priors_init["amplitude"] = prior_structure(
            #     halfnorm_ppf,
            #     loc=0.0,
            #     scale=A_guess - A0_guess,
            # )
            self.priors_init["theta"] = prior_structure(
                bounded_flat,
                low=-np.pi / 2,
                high=np.pi / 2,
                periodic=True,
            )
            self.priors_init["f"] = prior_structure(
                halfnorm_ppf,
                loc=0.0,
                scale=5.0,
            )
            self.priors_init["sigma"] = prior_structure(
                halfnorm_ppf,
                loc=0.0,
                scale=0.1,
            )
            self.priors_init["gamma"] = prior_structure(
                halfnorm_ppf,
                loc=0.2,
                scale=0.5,
            )
            self.priors_init["phi_0"] = prior_structure(
                bounded_flat,
                low=-np.pi,
                high=np.pi,
                periodic=True,
            )
            # self.priors_init["theta_gauss"] = prior_structure(
            #     # bounded_flat, low=0.0, high=np.pi, periodic=True,
            #     norm_ppf,
            #     mean=0.0,
            #     sigma=np.pi / 4.0,
            # )

            # ### nonlinearity parameters
            # self.priors_init["nl_alpha"] = prior_structure(
            #     halfnorm_ppf, loc=0.1, scale=1.0,
            # )
            # self.priors_init["nl_beta"] = prior_structure(
            #     halfnorm_ppf, loc=0.0, scale=1.0,
            # )
            # self.priors_init["nl_gamma"] = prior_structure(
            #     halfnorm_ppf, loc=0.0, scale=1.0,
            # )
        else:
            self.priors_init = priors_init

        super().set_priors(self.priors_init)

    def set_logp_func(
        self,
        vectorized=True,
        coding="simple",
        # penalties=["parameter_bias", "reliability", "overlap"]
    ):
        """
        TODO:
            instead of finding correlated trials before identification, run bayesian
            inference on all neurons, but adjust log-likelihood:

                * take placefield position, width, etc as hierarchical parameter
                    (narrow distribution for location and baseline activity?)
                * later, calculate logl for final parameter set to obtain active-trials (better logl)

            make sure all of this runs kinda fast!

            check, whether another hierarchy level should estimate noise-distribution parameters for overall data
                (thus, running inference only once on complete session, with N*4 parameters)
        """
        dx = self.X_FoV[0, 1] - self.X_FoV[0, 0]
        dy = self.Y_FoV[1, 0] - self.Y_FoV[0, 0]

        def get_logp(
            p_in,
            plot=False,
        ):

            self.timeit()
            """
                build switch between 3 models:
                    - gabor type model
                    - fourier component model (with n components in each direction)
                    - ellipse in rate space
            """

            if coding == "random":
                rate = np.zeros(self.dimensions["shape"])

            elif coding == "simple":
                params = self.get_params_from_p(p_in)
                self.timeit("transforming parameters")

                G = gabor_filter(self.X_FoV, self.Y_FoV, **params)
                self.timeit("calculating gabor filter")

                rate = np.einsum("ij,abcij->abc", G, self.gratings, order="C") * (
                    dx * dy
                )
                # rate = np.vdot(G, self.gratings) * dx * dy
                self.timeit("calculating model firing rate")

                # rate = (
                #     np.log(softplus(rate, gamma=2.0, delta=params["nl_offset"]))
                #     * params["nl_amplitude"]
                # )

            elif coding == "complex":
                params = self.get_params_from_p(p_in)
                self.timeit("transforming parameters")
                # print(params)
                G_even = gabor_filter(self.X_FoV, self.Y_FoV, **params)
                G_odd = gabor_filter(
                    self.X_FoV,
                    self.Y_FoV,
                    theta=params["theta"] + np.pi / 2,
                    f=params["f"],
                    sigma=params["sigma"],
                    gamma=params["gamma"],
                    phi_0=params["phi_0"],
                    theta_gauss=params.get("theta_gauss", None),
                )
                self.timeit("calculating gabor filter")

                rate_even = np.einsum(
                    "ij,abcij->abc", G_even, self.gratings, order="C"
                ) * (dx * dy)
                rate_odd = np.einsum(
                    "ij,abcij->abc", G_odd, self.gratings, order="C"
                ) * (dx * dy)
                self.timeit("calculating model firing rate")

                G = G_even
                rate = np.sqrt(rate_even**2 + rate_odd**2)

                # rate = (
                #     np.log(softplus(rate, gamma=2.0, delta=params["nl_offset"]))
                #     * params["nl_amplitude"]
                # )

            rate = softplus(rate * params["nl_amplitude"], delta=params["nl_offset"])

            ## no negative firing rates!!!
            self.timeit("applying nonlinearity")

            logp = self.probability_of_spike_observation(rate)
            self.timeit("calculating log probability of spike observation")

            if plot:
                self.plot_logp(G, rate, logp)

            # return logp.sum(axis=(1, 2, 3))
            if vectorized:
                return logp.sum(axis=(0, 1, 2))
            else:
                return logp.sum(axis=(0, 1, 2))

        return get_logp

    def probability_of_spike_observation(self, fmap_model):
        ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
        logp = (
            self.data["event_counts"] * np.log(fmap_model * self.data["T"])
            - self.log_N_factorial
            - fmap_model * self.data["T"]
        )

        # logp[np.logical_and(fmap_model == 0, self.data["event_counts"] == 0)] = 0
        logp[np.isnan(logp)] = -100.0
        # logp[np.isinf(logp)] = -100.0  # np.finfo(logp.dtype).min
        return logp

    def calculate_logp_penalty(self):
        """
        calculates several penalty values for the log-likelihood to adjust inference
            - zeroing_penalty: penalizes parameters to be far from 0

            - centering_penalty: penalizes parameters to be far from meta-parameter

            - activation_penalty: penalizes trials to be place-coding
                    This could maybe introduce AIC as penalty factor, to bias towards non-coding, and only consider it to be better, when it surpasses AIC? (how to properly implement?)

        """

        return None

    def plot_logp(self, G, rate, logp):

        # print(rate.shape)
        fig = plt.figure(figsize=(12, 8))
        print("PLOT PROBABILITY OVER COMPLETE PHASE SPACE")
        # idx_phase, idx_angle, idx_cycle = 2, 3, 9

        ## plot example of grating, gabor filter and neuron response
        # ax = fig.add_subplot(221)
        # ax.imshow(
        #     self.gratings[idx_phase, idx_angle, idx_cycle],
        #     cmap="gray",
        #     origin="lower",
        #     aspect="auto",
        # )
        # ax.set_title("Grating")

        X, Y = np.meshgrid(self.measure_points[1], self.measure_points[2])

        ax = fig.add_subplot(141, projection="3d")
        ax.plot_surface(
            self.X_FoV, self.Y_FoV, G, cmap="viridis"
        )  # , origin="lower", aspect="auto")

        ax.set_title("Gabor filter")

        fmap = self.data["event_counts"] / self.data["T"]
        # clims =

        options = {
            # "clim": np.percentile(fmap, [20, 80]),
            "extent": [
                *self.measure_points[1][[0, -1]],
                *self.measure_points[2][[0, -1]],
            ],
            "aspect": "auto",
            "origin": "lower",
            "cmap": "gray",
        }

        for idx_phase in range(self.dims["n_phases"]):
            # ax = fig.add_subplot(2, 4, 2 + idx_phase)
            ax = fig.add_subplot(3, 4, 2 + idx_phase, projection="3d")
            # Z = rate[idx_phase, ...].T
            # sigma = (1.0, 1.0)
            sigma = 0
            Z = gauss_smooth(rate[idx_phase, ...], sigma).T
            img = ax.plot_surface(
                X, Y, Z, cmap="viridis", linewidth=0, antialiased=True
            )
            ax.set_xlabel("angle")
            ax.set_ylabel("cycle")
            ax.set_zlabel("rate")
            ax.view_init(elev=30, azim=-60)

            ax.set_title("model response")
            # plt.colorbar(img)

            ax_emp = fig.add_subplot(3, 4, 6 + idx_phase, projection="3d")
            sigma = (1.0, 1.0)
            Z = gauss_smooth(fmap[idx_phase, ...], sigma).T
            # Z = fmap[idx_phase, ...].T

            # Z = Z.T

            img = ax_emp.plot_surface(
                X, Y, Z, cmap="viridis", linewidth=0, antialiased=True
            )
            ax_emp.set_xlabel("angle")
            ax_emp.set_ylabel("cycle")
            ax_emp.set_zlabel("rate")
            ax_emp.view_init(elev=30, azim=-60)

            # img = ax.imshow(
            #     gauss_smooth(
            #         fmap[idx_phase, ...],
            #         (1.0, 1.0),
            #     ).T,
            #     **options,
            # )
            ax_emp.set_title("empirical response")
            # plt.colorbar(img)
            plt.setp(ax, zlim=ax_emp.get_zlim())
            # plt.setp(ax, zlim=[6, 12])
            # plt.setp(ax_emp, zlim=[6, 12])

            ax = fig.add_subplot(3, 4, 10 + idx_phase, projection="3d")
            Z = gauss_smooth(logp[idx_phase, ...]).T

            img = ax.plot_surface(
                X, Y, Z, cmap="viridis", linewidth=0, antialiased=True
            )
            ax.set_title("logp")

        plt.tight_layout()


def sigmoid(x, alpha, beta, theta):
    return alpha / (1 + np.exp(-(x - theta) / beta))


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (np.sqrt(2) * sigma)))


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
