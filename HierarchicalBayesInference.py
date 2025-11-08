import logging, time, os, warnings, signal
import numpy as np
from scipy.special import factorial as sp_factorial, erf
from scipy.ndimage import gaussian_filter1d as gauss_filter
from scipy.interpolate import interp1d
import itertools

from .HierarchicalModelDefinition import HierarchicalModel
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

# from .utils import circmean as weighted_circmean, model_of_tuning_curve
# from .analyze_results import build_inference_results

os.environ["OMP_NUM_THREADS"] = "1"
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")

logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


class HierarchicalBayesInference(HierarchicalModel):

    def __init__(self, N, dwelltime, measure_points, FoV_range, logLevel=logging.ERROR):

        self.N = N  # [np.newaxis, ...]
        self.dwelltime = dwelltime  # [np.newaxis, ...]

        self.n_phases, self.n_angles, self.n_cycles = N.shape

        self.measure_points = measure_points
        self.grids = np.meshgrid(*measure_points)
        grid_phases, grid_angles, grid_cycles = self.grids

        # self.measure_grid = np.dstack((X, Y))
        # FoV_range = [val * np.pi / 180 for val in FoV_range]
        print(FoV_range)
        self.FoV_steps = 51
        self.X_FoV, self.Y_FoV = np.meshgrid(
            np.linspace(-FoV_range[0] / 2, FoV_range[0] / 2, self.FoV_steps),
            np.linspace(-FoV_range[1] / 2, FoV_range[1] / 2, self.FoV_steps),
        )
        self.FoV_grid = np.dstack((self.X_FoV, self.Y_FoV))
        self.X_FoV = self.X_FoV[np.newaxis, ...]
        self.Y_FoV = self.Y_FoV[np.newaxis, ...]

        self.gratings = self.get_gratings()

        # self.x_arr = np.arange(self.nbin)[np.newaxis, np.newaxis, :]

        self.firing_map = N.sum(axis=0) / dwelltime.sum(axis=0)

        ## pre-calculate log-factorial for speedup
        self.log_N_factorial = np.log(sp_factorial(self.N))

        self.log = logging.getLogger("nestLogger")
        self.log.setLevel(logLevel)

    def get_gratings(self):
        print("get gratings")
        gratings = np.zeros(
            (
                self.n_phases,
                self.n_angles,
                self.n_cycles,
                self.FoV_steps,
                self.FoV_steps,
            )
        )
        for prod in itertools.product(
            enumerate(self.measure_points[0]),
            enumerate(self.measure_points[1]),
            enumerate(self.measure_points[2]),
        ):
            idx, elems = zip(*prod)
            phi_0, theta, f = elems

            gratings[*idx, ...] = sine_grating(
                self.X_FoV[0, ...],
                self.Y_FoV[0, ...],
                theta * np.pi / 180,
                f,
                phi_0 * np.pi / 180,
                square=True,
            )
        return gratings

    def set_priors(self, priors_init=None):

        if priors_init is None:

            self.priors_init = {}
            self.priors_init["amplitude"] = prior_structure(
                halfnorm_ppf, loc=0.,scale=1.,
            )
            self.priors_init["sigma_x"] = prior_structure(
                halfnorm_ppf, loc=1.0, scale=10.0,
            )
            self.priors_init["sigma_y"] = prior_structure(
                halfnorm_ppf, loc=1.0, scale=10.0,
            )
            self.priors_init["gamma"] = prior_structure(
                halfnorm_ppf, loc=0.0, scale=1.0,
            )
            self.priors_init["f"] = prior_structure(
                halfnorm_ppf, loc=0.0, scale=0.2,
            )
            self.priors_init["theta"] = prior_structure(
                bounded_flat, low=0.0, high=np.pi, periodic=True,
            )
            self.priors_init["phi_0"] = prior_structure(
                bounded_flat, low=0.0, high=2 * np.pi, periodic=True,
            )
            self.priors_init["theta_gauss"] = prior_structure(
                bounded_flat, low=0.0, high=np.pi, periodic=True,
            )
            ### nonlinearity parameters
            self.priors_init["nl_alpha"] = prior_structure(
                halfnorm_ppf, loc=0.1, scale=1.0,
            )
            self.priors_init["nl_beta"] = prior_structure(
                halfnorm_ppf, loc=0.0, scale=1.0,
            )
            self.priors_init["nl_gamma"] = prior_structure(
                halfnorm_ppf, loc=0.0, scale=1.0,
            )
        else:
            self.priors_init = priors_init

        super().set_priors(self.priors_init)

    def model_of_gabor_filter(self, params):

        # Rotate coordinates
        x_prime = self.X_FoV * np.cos(params["theta"]) + self.Y_FoV * np.sin(
            params["theta"]
        )
        y_prime = -self.X_FoV * np.sin(params["theta"]) + self.Y_FoV * np.cos(
            params["theta"]
        )

        # print(f"{theta=}, {theta_gauss=}")
        if np.any(params["theta_gauss"]):
            x_prime_gauss = x_prime * np.cos(params["theta_gauss"]) + y_prime * np.sin(
                params["theta_gauss"]
            )
            y_prime_gauss = -x_prime * np.sin(params["theta_gauss"]) + y_prime * np.cos(
                params["theta_gauss"]
            )
        else:
            x_prime_gauss = x_prime
            y_prime_gauss = y_prime

        # Gabor function
        return (
            params["amplitude"]
            * np.exp(
                -(
                    x_prime_gauss**2 / (2 * params["sigma_x"] ** 2)
                    + params["gamma"] ** 2
                    * y_prime_gauss**2
                    / (2 * params["sigma_y"] ** 2)
                )
            )
            * np.cos(2 * np.pi * params["f"] * x_prime + params["phi_0"])
        )

    def from_p_to_params(self, p_in):
        """
        transform p_in to parameters for the model
        """
        params = {}
        if len(p_in.shape) == 1:
            p_in = p_in[np.newaxis, :]
        # if self.N_f > 0:
        #     params["PF"] = []
        #     for _ in range(self.N_f):
        #         params["PF"].append({})

        for key in self.priors:

            # if key.startswith("PF"):
            #     nField, key_param = key.split("_")
            #     nField = int(nField[2:]) - 1
            #     params["PF"][nField][key_param] = p_in[
            #         :,
            #         self.priors[key]["idx"] : self.priors[key]["idx"]
            #         + self.priors[key]["n"],
            #     ]
            # else:
            key_param = key.split("__")[0]
            params[key_param] = p_in[:, self.priors[key]["idx"]]
            if key_param.startswith("nl_"):
                params[key_param] = params[key_param][
                    :, np.newaxis, np.newaxis, np.newaxis
                ]
            else:
                params[key_param] = params[key_param][:, np.newaxis, np.newaxis]
            # : self.priors[key]["idx"]
            #     + self.priors[key]["n"],
            # ]

        return params

    # def from_results_to_params(self, results=None):
    #     """
    #     transform results to parameters for the model
    #     """
    #     results = results or self.inference_results
    #     params = {}

    #     if self.N_f > 0:
    #         params["PF"] = []
    #         for _ in range(self.N_f):
    #             params["PF"].append({})

    #     params["A0"] = results["fields"]["parameter"]["global"]["A0"][np.newaxis, 0]
    #     for key in ["theta", "A", "sigma"]:
    #         for f in range(self.N_f):
    #             if results["fields"]["parameter"]["local"][key] is None:
    #                 params["PF"][f][key] = results["fields"]["parameter"]["global"][
    #                     key
    #                 ][np.newaxis, f, 0]
    #             else:
    #                 params["PF"][f][key] = results["fields"]["parameter"]["local"][key][
    #                     np.newaxis, f, :, 0
    #                 ]

    #     return params

    def timeit(self, msg=None):
        if not msg is None:  # and (self.time_ref):
            self.log.debug(f"time for {msg}: {(time.time()-self.time_ref)*10**6}")

        self.time_ref = time.time()

    def set_logp_func(
        self,
        vectorized=True,
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

        import matplotlib.pyplot as plt

        def get_logp(
            p_in,
            # get_active_model=False,
            # get_logp=False,
            # get_tuning_curve=False,
            plot=False,
        ):

            # t_ref = time.time()
            ## adjust shape of incoming parameters to vectorized analysis
            # if len(p_in.shape) == 1:
            #     p_in = p_in[np.newaxis, :]
            # N_in = p_in.shape[0]

            self.timeit()

            params = self.from_p_to_params(p_in)
            # print(params)
            self.timeit("transforming parameters")

            G = self.model_of_gabor_filter(params)

            # print(G.shape, self.gratings.shape)
            # fmap_model = (G * self.gratings).sum(axis=(-2, -1))
            fmap_model = np.tensordot(G, self.gratings, axes=([1, 2], [3, 4]))
            # print(fmap_model.shape)
            rate = softplus(
                fmap_model,
                params["nl_alpha"],
                params["nl_beta"],
                params["nl_gamma"],
                # fmap_model,
                # params["nl_alpha"],
                # 0,
                # 0,
            )

            logp = self.probability_of_spike_observation(rate)
            # print(logp)
            # logp = self.probability_of_spike_observation(fmap_model)

            if plot:
                # print(rate.shape)
                fig = plt.figure()

                idx_phase, idx_angle, idx_cycle = 0, 3, 9

                ## plot example of grating, gabor filter and neuron response
                ax = fig.add_subplot(221)
                ax.imshow(
                    self.gratings[idx_phase, idx_angle, idx_cycle],
                    cmap="gray",
                    origin="lower",
                    aspect="auto",
                )
                ax.set_title("Grating")

                ax = fig.add_subplot(222)
                ax.imshow(G[0, ...], cmap="gray")
                ax.set_title("Gabor filter")

                ax = fig.add_subplot(223)
                img = ax.imshow(
                    fmap_model[0, idx_phase, ...],
                    extent=[
                        *self.measure_points[1][[0, -1]],
                        *self.measure_points[2][[0, -1]],
                    ],
                    cmap="gray",
                    origin="lower",
                    aspect="auto",
                )
                ax.set_title("Neuron response")
                plt.colorbar(img)

                ax = fig.add_subplot(224)
                img = ax.imshow(
                    rate[0, idx_phase, ...],
                    extent=[
                        *self.measure_points[1][[0, -1]],
                        *self.measure_points[2][[0, -1]],
                    ],
                    cmap="gray",
                    origin="lower",
                    aspect="auto",
                )
                ax.set_title("Neuron response")
                plt.colorbar(img)

                plt.tight_layout()

            # print(rate)

            ## no negative firing rates!!!
            # fmap_model = np.clip(fmap_model, a_min=0.01, a_max=np.inf)
            # print(fmap_model.shape)

            # print(logp.shape)

            return logp.sum(axis=(1, 2, 3))
            # if vectorized:
            #     return logp
            # else:
            #     return logp[0]

        return get_logp

    def probability_of_spike_observation(self, fmap_model):
        ## get probability to observe N spikes (amplitude) within dwelltime for each bin in each trial
        logp = (
            self.N * np.log(fmap_model * self.dwelltime)
            - self.log_N_factorial
            - fmap_model * self.dwelltime
        )

        # logp[np.logical_and(fmap_model == 0, self.N == 0)] = 0
        logp[np.isnan(logp)] = -100.0
        logp[np.isinf(logp)] = -100.0  # np.finfo(logp.dtype).min
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

    # def model_comparison(
    #     self,
    #     hierarchical=["theta"],
    #     wrap=[],
    #     show_status=False,
    #     limit_execution_time=None,
    # ):
    #     t_start = time.time()
    #     self.inference_results = build_inference_results(
    #         N_f=2,
    #         nbin=self.nbin,
    #         mode="bayesian",
    #         n_trials=self.nSamples,
    #         hierarchical=hierarchical,
    #     )

    #     if limit_execution_time:

    #         def handler(signum, frame):
    #             print("Forever is over!")
    #             raise TimeoutException("end of time")

    #         signal.signal(signal.SIGALRM, handler)

    #     if (self.N > 0).sum() > 10:
    #         # if (activity[self.behavior["active"]] > 0).sum() < 10:
    #         # print("Not enough instances of activity detected")
    #         # return None

    #         previous_logz = -np.inf
    #         for f in range(2 + 1):
    #             if show_status:
    #                 print(f"\n{f=}\n")

    #             try:
    #                 if limit_execution_time:
    #                     signal.alarm(limit_execution_time)
    #                 self.set_priors(N_f=f, hierarchical_in=hierarchical, wrap=wrap)

    #                 sampling_results = self.run_sampling(
    #                     penalties=["overlap", "reliability"],
    #                     improvement_loops=2,
    #                     show_status=show_status,
    #                 )

    #                 if limit_execution_time:
    #                     signal.alarm(0)

    #                 self.inference_results["fields"]["logz"][f, 0] = sampling_results[
    #                     "logz"
    #                 ]
    #                 self.inference_results["fields"]["logz"][f, 1] = sampling_results[
    #                     "logzerr"
    #                 ]
    #                 if previous_logz > sampling_results["logz"]:
    #                     self.N_f = f - 1
    #                     break

    #                 self.store_inference_results(sampling_results)
    #                 previous_logz = sampling_results["logz"]
    #             except Exception as exc:
    #                 print("Exception:", exc)
    #                 break

    #         self.calculate_general_statistics()

    #         # if plot:
    #         #     self.display_results()

    #         string_out = f"Model comparison finished after {time.time() - t_start:.2f}s with evidences: "
    #         for f in range(2 + 1):
    #             if not np.isnan(self.inference_results["fields"]["logz"][f, 0]):
    #                 string_out += f"\t {f=} {'*' if f==self.inference_results['fields']['n_modes'] else ''}, logz={self.inference_results['fields']['logz'][f,0]:.2f}"
    #     else:
    #         self.calculate_general_statistics(which=["firingstats"])

    #         string_out = "Not enough instances of activity detected"

    #     print(string_out)

    #     return self.inference_results


def sine_grating(X, Y, theta, f, phi_0, square=False):
    # Rotate coordinates
    x_prime = X * np.cos(theta) + Y * np.sin(theta)

    # Gabor function
    if square:
        return np.sign(np.cos(2 * np.pi * f * x_prime + phi_0))
    else:
        return np.cos(2 * np.pi * f * x_prime + phi_0)


# def softplus(x, alpha, beta, gamma, delta):
def softplus(x, alpha, beta, gamma):
    # return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta
    return alpha * np.log(1 + np.exp((x - gamma) / beta))  # + delta


def ReLU(x, alpha, beta, theta):
    return alpha * np.maximum(beta, x - theta)


def sigmoid(x, alpha, beta, theta):
    return alpha / (1 + np.exp(-(x - theta) / beta))


def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (np.sqrt(2) * sigma)))


class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
