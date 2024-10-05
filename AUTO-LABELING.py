import seisbench.data as sbd
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ex
import plotly.graph_objects as go
from obspy.clients.fdsn.header import FDSNException
from pathlib import Path
from obspy.core.event import read_events
import scipy.signal as signal
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xml.etree.ElementTree as ET

client = Client('IRIS')

data = sbd.WaveformDataset('data' , cache = 'full')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass Butterworth filter to the input data.

    Parameters:
    - data: Input signal (1D array)
    - lowcut: Low cutoff frequency for the bandpass filter in Hz
    - highcut: High cutoff frequency for the bandpass filter in Hz
    - fs: Sampling frequency in Hz
    - order: The order of the filter (default is 4)

    Returns:
    - filtered_data: Bandpass-filtered signal (1D array)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    print(low,high)
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter using filtfilt to prevent phase distortion
    filtered_data = filtfilt(b, a, data)

    return filtered_data
def highpass_filter(data, cutoff, fs, order=4):
    """
    Applies a highpass Butterworth filter to the input data.

    Parameters:
    - data: Input signal (1D array)
    - cutoff: Cutoff frequency for the highpass filter in Hz
    - fs: Sampling frequency in Hz
    - order: The order of the filter (default is 4)

    Returns:
    - filtered_data: Highpass-filtered signal (1D array)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist

    # Design Butterworth highpass filter
    b, a = butter(order, normal_cutoff, btype='high')

    # Apply the filter using filtfilt to prevent phase distortion
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def get_filtered(sig,highpass_cut=1, bandpass_lowcut= .1,bandpass_highcut =1.5, band_filt = True,high_filt=True):

    # Apply the high-pass filter to remove frequencies below 0.5 Hz
    filtered_trace = signal.detrend(sig)
    filtered_trace = signal.medfilt(filtered_trace, kernel_size=21)
    if high_filt:
        filtered_trace = highpass_filter(sig, highpass_cut, fs=20)
    if band_filt:
        filtered_trace = bandpass_filter(sig, bandpass_lowcut, bandpass_highcut,fs=20)


    return filtered_trace

for i in range(0,len(data)):

    try:
        number_trace = i  # iterate on this

        filtered_trace0 = get_filtered(data.get_waveforms(number_trace)[0].T,
                                       highpass_cut=.5,
                                       # bandpass_lowcut= .01,
                                       # bandpass_highcut =1,
                                       band_filt=False)
        frequencies, times, Sxx_0 = signal.spectrogram(filtered_trace0, 20, nfft=2000, nperseg=400, noverlap=200, scaling='density')

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Hide axes and spines
        ax.axis('off')

        # Plot the spectrogram
        pcm = ax.pcolormesh(times, frequencies, Sxx_0, cmap='viridis', shading='gouraud')

        ymin, ymax = ax.get_ylim()

        # Define the bounding box dimensions in data units
        bbox_x = data.metadata.iloc[number_trace]['trace_P_spectral_start_arrival_sample'] / 20
        bbox_width = data.metadata.iloc[number_trace]['trace_S_spectral_end_arrival_sample'] / 20 - bbox_x
        bbox_y = ymin
        bbox_height = ymax - ymin

        # Create a Rectangle patch in data units
        rect = Rectangle((bbox_x, bbox_y), bbox_width, bbox_height,
                         linewidth=2, edgecolor='red', facecolor='none', alpha=0.6)
        # ax.add_patch(rect)
        plt.savefig(f'plot{i}.png', dpi=100)
        fig_width_px = fig.get_figwidth() * fig.dpi
        fig_height_px = fig.get_figheight() * fig.dpi

        # Get the axis limits in data coordinates
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        min_x_disp = int(((rect.get_x() - xmin) / (xmax - xmin)) * fig_width_px)
        max_x_disp = int(((rect.get_x() + rect.get_width()) - xmin) / (xmax - xmin) * fig_width_px)
        min_y_disp = int(((rect.get_y() - ymin) / (ymax - ymin)) * fig_height_px)
        max_y_disp = int(((rect.get_y() + rect.get_height()) - ymin) / (ymax - ymin) * fig_height_px)
        X_center = (min_x_disp + max_x_disp) / 2
        Y_center = (min_y_disp + max_y_disp) / 2
        x_width = max_x_disp - min_x_disp
        y_height = max_y_disp - min_y_disp

        # Normalize the values (0 to 1 range for YOLO format)
        X_center_norm = X_center / fig_width_px
        Y_center_norm = Y_center / fig_height_px
        x_width_norm = x_width / fig_width_px
        y_height_norm = y_height / fig_height_px

        # YOLO formatted output
        coordinate = "1 {:.6f} {:.6f} {:.6f} {:.6f}".format(X_center_norm, Y_center_norm, x_width_norm, y_height_norm)

        # plot{i}.txt
        with open(f'plot{i}.txt', 'w') as f:
            f.write(coordinate)
    except Exception as e:
        print(f"An error occurred while processing trace {i}: {e}")


