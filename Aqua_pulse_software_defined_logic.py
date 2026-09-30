import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate


# ===============================================================
# CONFIGURATION
# ===============================================================

FS = 850_000  # DAC/sample rate = 850 kHz

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================================
# 1. SINE LOOKUP TABLE
#    10-bit index + Q15 amplitude
# ===============================================================

LUT_BITS = 10
LUT_SIZE = 1 << LUT_BITS

sine_lut = np.round(
    32767
    * np.sin(
        2 * np.pi * np.arange(LUT_SIZE + 1) / LUT_SIZE
    )
).astype(np.int32)


def lut_sin(phase_u32):
    """
    32-bit phase accumulator -> interpolated Q15 sine.

    This approximates the type of phase-accumulator/LUT
    implementation that can be used on an STM32.
    """

    phase_u32 = int(phase_u32) & 0xFFFFFFFF

    # LUT index
    idx = (
        phase_u32 >> (32 - LUT_BITS)
    ) & (LUT_SIZE - 1)

    # 16-bit interpolation fraction
    frac = (
        phase_u32 >> (32 - LUT_BITS - 16)
    ) & 0xFFFF

    s0 = int(sine_lut[idx])
    s1 = int(sine_lut[idx + 1])

    # Linear interpolation
    return s0 + (((s1 - s0) * frac) >> 16)


# ===============================================================
# 2. WINDOW TABLES
#    Returned as Q15 integer values
# ===============================================================

def window_table(kind, n):

    x = np.linspace(
        0,
        1,
        n,
        endpoint=False
    )

    if kind == "rect":

        w = np.ones(n)

    elif kind == "hann":

        w = (
            0.5
            - 0.5 * np.cos(2 * np.pi * x)
        )

    elif kind == "hamming":

        w = (
            0.54
            - 0.46 * np.cos(2 * np.pi * x)
        )

    elif kind == "blackman":

        w = (
            0.42
            - 0.5 * np.cos(2 * np.pi * x)
            + 0.08 * np.cos(4 * np.pi * x)
        )

    elif kind == "tukey":

        alpha = 0.25

        w = np.ones(n)

        taper = int(alpha * n / 2)

        if taper > 0:

            t = np.linspace(
                0,
                np.pi,
                taper,
                endpoint=False
            )

            w[:taper] = (
                0.5
                * (1 - np.cos(t))
            )

            w[-taper:] = w[:taper][::-1]

    else:

        raise ValueError(
            f"Unknown window type: {kind}"
        )

    return np.round(
        w * 32767
    ).astype(np.int32)


# ===============================================================
# 3. BARKER-13 PHASE CODE
# ===============================================================

BARKER13 = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1
    ],
    dtype=np.int32
)


# ===============================================================
# 4. FIXED-POINT PHASE ACCUMULATOR CONSTANTS
# ===============================================================

PHASE_BITS = 32
PHASE_MOD = 1 << PHASE_BITS


def frequency_to_ftw(freq, fs):
    """
    Convert frequency to a 32-bit DDS frequency tuning word.

    FTW = frequency / sample_rate * 2^32
    """

    return int(
        round(
            (freq / fs) * PHASE_MOD
        )
    )


# ===============================================================
# 5. LFM CHIRP
# ===============================================================

def synth_lfm(
    fs,
    f0,
    f1,
    T,
    amplitude,
    win
):

    N = int(round(fs * T))

    buf = np.zeros(
        N,
        dtype=np.int32
    )

    # DAC is assumed to be 12-bit
    # 0 ... 4095
    DAC_MID = 2048

    A_DAC = int(
        round(
            amplitude * 2047
        )
    )

    # Initial frequency tuning word
    ftw = frequency_to_ftw(
        f0,
        fs
    )

    # Frequency increment per sample
    #
    # df/dt = (f1-f0)/T
    #
    # FTW increment:
    #
    # dFTW = df/dt / fs * 2^32
    #
    dftw = int(
        round(
            (
                (f1 - f0)
                / (T * fs)
            )
            * PHASE_MOD
            / fs
        )
    )

    # Equivalent expression:
    #
    # dftw = ((f1-f0) / (T * fs^2)) * 2^32

    phase = 0

    for n in range(N):

        s = lut_sin(phase)

        # Apply window
        s_windowed = (
            s * int(win[n])
        ) >> 15

        # Apply amplitude
        output = (
            s_windowed * A_DAC
        ) >> 15

        buf[n] = (
            DAC_MID + output
        )

        # Update phase
        phase = (
            phase + ftw
        ) & 0xFFFFFFFF

        # Update frequency
        ftw = (
            ftw + dftw
        ) & 0xFFFFFFFF

    return buf


# ===============================================================
# 6. GEOMETRIC / LOGARITHMIC CHIRP
# ===============================================================

def synth_geometric(
    fs,
    f0,
    f1,
    T,
    amplitude,
    win
):

    N = int(round(fs * T))

    buf = np.zeros(
        N,
        dtype=np.int32
    )

    DAC_MID = 2048

    A_DAC = int(
        round(
            amplitude * 2047
        )
    )

    # Use floating point only for calculating
    # the geometric frequency ratio.
    #
    # The actual phase accumulator remains
    # integer based.

    ratio = (
        f1 / f0
    ) ** (1.0 / N)

    frequency = f0

    phase = 0

    for n in range(N):

        # Calculate current FTW
        ftw = frequency_to_ftw(
            frequency,
            fs
        )

        s = lut_sin(phase)

        # Window
        s_windowed = (
            s * int(win[n])
        ) >> 15

        # Amplitude
        output = (
            s_windowed * A_DAC
        ) >> 15

        buf[n] = (
            DAC_MID + output
        )

        # Update phase
        phase = (
            phase + ftw
        ) & 0xFFFFFFFF

        # Geometric frequency progression
        frequency *= ratio

        # Avoid numerical drift above f1
        if frequency > f1:
            frequency = f1

    return buf


# ===============================================================
# 7. BARKER CODED WAVEFORM
# ===============================================================

def synth_coded(
    fs,
    f0,
    f1,
    T,
    amplitude,
    win
):

    N = int(round(fs * T))

    buf = np.zeros(
        N,
        dtype=np.int32
    )

    DAC_MID = 2048

    A_DAC = int(
        round(
            amplitude * 2047
        )
    )

    # Carrier frequency
    fc = 0.5 * (
        f0 + f1
    )

    ftw = frequency_to_ftw(
        fc,
        fs
    )

    phase = 0

    # Number of samples per Barker chip
    chip_len = max(
        1,
        N // len(BARKER13)
    )

    for n in range(N):

        chip = min(
            n // chip_len,
            len(BARKER13) - 1
        )

        # +1 -> normal phase
        # -1 -> phase shifted by 180 degrees

        if BARKER13[chip] > 0:
            phase_offset = 0
        else:
            phase_offset = 1 << 31

        coded_phase = (
            phase + phase_offset
        ) & 0xFFFFFFFF

        s = lut_sin(
            coded_phase
        )

        # Window
        s_windowed = (
            s * int(win[n])
        ) >> 15

        # Amplitude
        output = (
            s_windowed * A_DAC
        ) >> 15

        buf[n] = (
            DAC_MID + output
        )

        phase = (
            phase + ftw
        ) & 0xFFFFFFFF

    return buf


# ===============================================================
# 8. MAIN SYNTHESIS FUNCTION
# ===============================================================

def synth_ping(
    mode,
    fs,
    f0,
    f1,
    T,
    amplitude,
    window="hann"
):

    N = int(
        round(
            fs * T
        )
    )

    win = window_table(
        window,
        N
    )

    if mode == "lfm":

        buf = synth_lfm(
            fs=fs,
            f0=f0,
            f1=f1,
            T=T,
            amplitude=amplitude,
            win=win
        )

    elif mode == "geometric":

        buf = synth_geometric(
            fs=fs,
            f0=f0,
            f1=f1,
            T=T,
            amplitude=amplitude,
            win=win
        )

    elif mode == "coded":

        buf = synth_coded(
            fs=fs,
            f0=f0,
            f1=f1,
            T=T,
            amplitude=amplitude,
            win=win
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return buf, N


# ===============================================================
# 9. SONAR SCENARIOS
# ===============================================================

scenarios = {

    # Clear water / reef
    "clear_reef_lfm": {

        "mode": "lfm",
        "f0": 30_000,
        "f1": 130_000,
        "T": 1.0e-3,
        "amplitude": 0.44
    },

    # Muddy / turbid water
    "muddy_estuary_lfm": {

        "mode": "lfm",
        "f0": 21_000,
        "f1": 27_000,
        "T": 10.0e-3,
        "amplitude": 0.86
    },

    # Geometric chirp
    "clear_reef_geom": {

        "mode": "geometric",
        "f0": 30_000,
        "f1": 130_000,
        "T": 1.0e-3,
        "amplitude": 0.44
    },

    # Barker-coded waveform
    "muddy_estuary_coded": {

        "mode": "coded",
        "f0": 21_000,
        "f1": 27_000,
        "T": 10.0e-3,
        "amplitude": 0.86
    }
}


# ===============================================================
# 10. ANALYSIS FUNCTIONS
# ===============================================================

def calculate_peak_to_peak(sig):

    return (
        np.max(sig)
        - np.min(sig)
    )


def calculate_rms(sig):

    return np.sqrt(
        np.mean(
            sig ** 2
        )
    )


def calculate_autocorrelation(sig):

    corr = correlate(
        sig,
        sig,
        mode="full"
    )

    max_value = np.max(
        np.abs(corr)
    )

    if max_value != 0:

        corr = (
            corr / max_value
        )

    return corr


# ===============================================================
# 11. RUN SIMULATION
# ===============================================================

print("=" * 70)
print("ADAPTIVE SOFTWARE-DEFINED SONAR SIMULATION")
print("=" * 70)

print(
    f"Sample rate : {FS / 1000:.1f} kHz"
)

print(
    f"Output dir  : {os.path.abspath(OUTPUT_DIR)}"
)

print("=" * 70)


for name, cfg in scenarios.items():

    print()
    print(
        f"Processing: {name}"
    )

    # -----------------------------------------------------------
    # Generate waveform
    # -----------------------------------------------------------

    buf, N = synth_ping(
        fs=FS,
        window="hann",
        **cfg
    )

    # Time axis
    t = (
        np.arange(N)
        / FS
    )

    # Remove 12-bit DAC DC offset
    sig = (
        buf.astype(np.float64)
        - 2048.0
    )

    # -----------------------------------------------------------
    # Basic measurements
    # -----------------------------------------------------------

    duration_ms = (
        N / FS * 1000
    )

    peak = np.max(
        np.abs(sig)
    )

    peak_to_peak = (
        calculate_peak_to_peak(
            sig
        )
    )

    rms = calculate_rms(
        sig
    )

    # -----------------------------------------------------------
    # Autocorrelation / matched-filter-like response
    # -----------------------------------------------------------

    corr = calculate_autocorrelation(
        sig
    )

    lag = (
        np.arange(
            len(corr)
        )
        - (len(corr) - 1) // 2
    ) / FS * 1000

    # -----------------------------------------------------------
    # Spectrogram
    # -----------------------------------------------------------

    # Select reasonable STFT parameters
    nperseg = min(
        1024,
        max(128, N // 8)
    )

    noverlap = int(
        nperseg * 0.75
    )

    f, tt, Sxx = spectrogram(
        sig,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    Sxx_dB = (
        10
        * np.log10(
            Sxx + 1e-12
        )
    )

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(10, 10)
    )

    fig.suptitle(
        f"{name}\n"
        f"Mode = {cfg['mode']} | "
        f"f0 = {cfg['f0']/1000:.1f} kHz | "
        f"f1 = {cfg['f1']/1000:.1f} kHz | "
        f"T = {cfg['T']*1000:.2f} ms",
        fontsize=13
    )

    # ===========================================================
    # Plot 1: Time domain
    # ===========================================================

    ax[0].plot(
        t * 1000,
        sig,
        linewidth=0.8
    )

    ax[0].set_xlabel(
        "Time (ms)"
    )

    ax[0].set_ylabel(
        "DAC code (AC)"
    )

    ax[0].set_title(
        "Time-Domain Waveform"
    )

    ax[0].grid(
        True,
        alpha=0.3
    )

    # ===========================================================
    # Plot 2: Spectrogram
    # ===========================================================

    mesh = ax[1].pcolormesh(
        tt * 1000,
        f / 1000,
        Sxx_dB,
        shading="auto"
    )

    ax[1].set_xlabel(
        "Time (ms)"
    )

    ax[1].set_ylabel(
        "Frequency (kHz)"
    )

    ax[1].set_title(
        "Spectrogram (STFT)"
    )

    ax[1].set_ylim(
        0,
        FS / 2 / 1000
    )

    fig.colorbar(
        mesh,
        ax=ax[1],
        label="Power (dB)"
    )

    # ===========================================================
    # Plot 3: Autocorrelation
    # ===========================================================

    ax[2].plot(
        lag,
        corr,
        linewidth=0.8
    )

    ax[2].set_xlabel(
        "Lag (ms)"
    )

    ax[2].set_ylabel(
        "Normalised correlation"
    )

    ax[2].set_title(
        "Autocorrelation / Pulse Compression Response"
    )

    ax[2].grid(
        True,
        alpha=0.3
    )

    # ===========================================================
    # Save figure
    # ===========================================================

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{name}.png"
    )

    fig.savefig(
        output_file,
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    # -----------------------------------------------------------
    # Print results
    # -----------------------------------------------------------

    print(
        f"  Mode          : {cfg['mode']}"
    )

    print(
        f"  Samples       : {N}"
    )

    print(
        f"  Duration      : {duration_ms:.3f} ms"
    )

    print(
        f"  f0            : {cfg['f0']/1000:.2f} kHz"
    )

    print(
        f"  f1            : {cfg['f1']/1000:.2f} kHz"
    )

    print(
        f"  Amplitude     : {cfg['amplitude']:.2f}"
    )

    print(
        f"  Peak          : {peak:.0f} DAC codes"
    )

    print(
        f"  Peak-to-peak  : {peak_to_peak:.0f} DAC codes"
    )

    print(
        f"  RMS           : {rms:.2f} DAC codes"
    )

    print(
        f"  Saved         : {output_file}"
    )


# ===============================================================
# 12. FINISHED
# ===============================================================

print()
print("=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

print(
    f"All waveform plots are saved in:"
)

print(
    os.path.abspath(OUTPUT_DIR)
)

print()
print(
    "Generated files:"
)

for name in scenarios:

    print(
        f"  - {name}.png"
    )

print("=" * 70)
