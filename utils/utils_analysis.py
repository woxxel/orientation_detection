import numpy as np
import itertools

from placefield_dynamics.placefield_detection.utils.utils_analysis import (
    get_spiking_data,
)


def get_spikes(
        S,
        f=30.05,
):
    spikes = np.zeros_like(S)
    for n, s in enumerate(S):
        spikes[n, :], _, _ = get_spiking_data(s, f=f)

    return spikes

def get_unique_stimulus_values(stimulus_data):
    unique_values = {
        "phases": np.unique(stimulus_data["stim_phases"]),
        "angles": np.unique(stimulus_data["stim_angle"]),
        "cycles": np.unique(stimulus_data["stim_cycles"]),
    }
    return unique_values


def calculate_firing_maps(
        stimulus_data, 
        spikes,
        # f=30.05,
        dt_onset = 0.25,  # time to wait after stimulus onset before starting to count spikes
        dt_response = 0.25
):
    n_cells = spikes.shape[0]

    unique_values = get_unique_stimulus_values(stimulus_data)

    event_counts = np.zeros(
        (
            len(unique_values["phases"]),
            len(unique_values["angles"]),
            len(unique_values["cycles"]),
            n_cells,
        )
    )

    dwelltime = np.zeros(
        (
            len(unique_values["phases"]),
            len(unique_values["angles"]),
            len(unique_values["cycles"]),
        )
    )

    for prod in itertools.product(
        enumerate(unique_values["phases"]),
        enumerate(unique_values["angles"]),
        enumerate(unique_values["cycles"]),
    ):
        idx, elems = zip(*prod)

        idxes = (
            (stimulus_data["stim_phases"] == elems[0])
            & (stimulus_data["stim_angle"] == elems[1])
            & (stimulus_data["stim_cycles"] == elems[2])
        ).flatten()

        # print(idx,elems)
        times = stimulus_data["stimulus_clock"][idxes, :]
        for t, time in enumerate(times):
            # print(time)
            start_idx = np.argmin(np.abs(stimulus_data["frame_times"] - (time[0] + dt_onset)))
            end_idx = np.argmin(
                np.abs(stimulus_data["frame_times"] - (time[0] + dt_onset + dt_response))
            )

            event_counts[idx[0], idx[1], idx[2], :] += spikes[:, start_idx:end_idx].sum(axis=1)
            # / (stimulus_data["frame_times"][end_idx] - stimulus_data["frame_times"][start_idx])

            dwelltime[idx[0], idx[1], idx[2]] += (
                stimulus_data["frame_times"][end_idx] - stimulus_data["frame_times"][start_idx]
            )

    return event_counts, dwelltime        
