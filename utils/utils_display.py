import numpy as np
import matplotlib.pyplot as plt

from placefield_dynamics.placefield_detection.utils import (
    gauss_smooth,
)  

def plot_stimulus(data):

    stimulus = ["stim_angle", "stim_cycles", "stim_phases"]

    fig = plt.figure(figsize=(8, 7))

    gs = fig.add_gridspec(5, 5)
    ax = fig.add_subplot(gs[0, :-1])
    ax.plot(data["grating"][0, :], ".")

    ax = fig.add_subplot(gs[0, -1])
    ax.hist(np.diff(data["stimulus_clock"][:, 0]), np.linspace(0.2, 0.3, 51))
    plt.setp(ax, xlabel="ISI")

    for i, stim in enumerate(stimulus):

        stim_data = data[stim].flatten()
        ax = fig.add_subplot(gs[i + 1, :-1])
        ax.plot(data["stimulus_clock"][:, 0], stim_data, ".", label=stim)
        plt.setp(ax, ylabel=stim)

        ax = fig.add_subplot(gs[i + 1, -1])
        ax.hist(
            stim_data, np.linspace(np.min(stim_data), np.max(stim_data), 51)
        )  # ,label=stim)
        plt.setp(ax, xlabel=stim)
    plt.tight_layout()


def plot_firing_maps(firing_maps, neurons, unique_values):

    # neurons = range(10)
    rows = len(neurons)

    fig = plt.figure(figsize=(10, rows * 2))

    cols = 4
    gs = fig.add_gridspec(rows, cols)

    sigma = (1.0, 1.0)
    options = {
        "extent": [
            *unique_values["angles"][[0,-1]],
            *unique_values["cycles"][[0,-1]],
        ],
        "aspect": "auto",
        "origin": "lower",
    }

    def plot_firing_map(ax, firing_map, sigma=(1.0, 1.0), **options):

        img = ax.imshow(
            gauss_smooth(firing_map, sigma),
            **options,
        )
        plt.colorbar(img)


    for row, n in enumerate(neurons):

        clims = np.percentile(firing_maps[..., n], [20, 80])
        for i, phase in enumerate(unique_values["phases"]):
            ax = fig.add_subplot(gs[row, i])
            plot_firing_map(ax, firing_maps[i, ..., n].T, sigma, **options, clim=clims)

            plt.setp(
                ax,
                title=f"Phase: {phase}°" if row == 0 else None,
                xlabel="Angle" if row == rows - 1 else None,
                ylabel="CPD",
            )

        ax = fig.add_subplot(gs[row, 3])
        plot_firing_map(
            ax, firing_maps[..., n].mean(axis=0).T, sigma, **options#, clim=clims
        )
        ax.text(0, 0.13, f"Neuron {n}")
        plt.setp(ax, title=f"mean response" if row == 0 else None)

    plt.tight_layout()