import sys
import queue
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import stft

# -----------------------------
# Settings you might tweak
# -----------------------------
DEVICE = None          # e.g. 0, 1, 2... or None for default input device
CHANNELS = 1
SAMPLERATE = 44100     # try 48000 if your mic prefers it
BLOCKSIZE = 1024       # frames per audio callback block
WINDOW_SECONDS = 2.0   # how many seconds of audio to display in the spectrogram

NFFT = 2048            # STFT FFT size
HOP = 256              # STFT hop size (smaller = smoother updates, more CPU)
FREQ_MAX = 8000        # show up to this frequency (Hz). set None to show full range

UPDATE_HZ = 30         # plot refresh rate
EPS = 1e-10            # to avoid log(0)

# -----------------------------
# Audio plumbing
# -----------------------------
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Called in a real-time audio thread; keep this very fast."""
    if status:
        # status can include underruns/overruns; printing is okay but can be noisy
        print(status, file=sys.stderr)

    # indata shape: (frames, channels)
    mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
    audio_q.put(mono)

def get_available_input_devices():
    devices = sd.query_devices()
    inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    return inputs

# -----------------------------
# Spectrogram logic
# -----------------------------
def compute_spectrogram(audio, fs):
    """
    Compute magnitude spectrogram in dB using STFT.
    Returns: (freqs, times, S_db)
    """
    f, t, Zxx = stft(
        audio,
        fs=fs,
        nperseg=NFFT,
        noverlap=NFFT - HOP,
        nfft=NFFT,
        window="hann",
        padded=False,
        boundary=None,
    )
    S = np.abs(Zxx)
    S_db = 20.0 * np.log10(S + EPS)
    return f, t, S_db

def main():
    # Helpful device list if something fails
    if DEVICE is None:
        try:
            default_in = sd.default.device[0]
            if default_in is None or default_in < 0:
                raise RuntimeError
        except Exception:
            print("No default input device found. Available input devices:")
            for i, name in get_available_input_devices():
                print(f"  {i}: {name}")
            return

    ring_len = int(SAMPLERATE * WINDOW_SECONDS)
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live Spectrogram (STFT)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Initialize with a dummy spectrogram so imshow exists
    dummy_audio = np.zeros(ring_len, dtype=np.float32)
    f, t, S_db = compute_spectrogram(dummy_audio, SAMPLERATE)

    if FREQ_MAX is not None:
        f_mask = f <= FREQ_MAX
        f_plot = f[f_mask]
        S_plot = S_db[f_mask, :]
    else:
        f_plot = f
        S_plot = S_db

    # Display range (dB). You can tweak these for contrast.
    vmin, vmax = -90, -20

    im = ax.imshow(
        S_plot,
        origin="lower",
        aspect="auto",
        extent=[-WINDOW_SECONDS, 0.0, f_plot[0], f_plot[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    fig.canvas.draw()
    fig.show()

    print("Starting audio stream... (Ctrl+C to stop)")
    try:
        with sd.InputStream(
            device=DEVICE,
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            callback=audio_callback,
        ):
            # Main UI loop
            last_update = 0.0
            update_period = 1.0 / float(UPDATE_HZ)

            while plt.fignum_exists(fig.number):
                # Drain audio queue
                while True:
                    try:
                        block = audio_q.get_nowait()
                    except queue.Empty:
                        break

                    n = len(block)
                    if n <= 0:
                        continue

                    # Write into ring buffer (circular)
                    end = write_pos + n
                    if end < ring_len:
                        ring[write_pos:end] = block
                    else:
                        first = ring_len - write_pos
                        ring[write_pos:] = block[:first]
                        ring[: end % ring_len] = block[first:]
                    write_pos = end % ring_len

                # Update at UPDATE_HZ
                now = plt.get_fignums()  # cheap keep-alive
                # Use plt.pause to give time back to UI & reduce CPU
                plt.pause(update_period)

                # Reconstruct audio in chronological order (oldest -> newest)
                audio = np.concatenate([ring[write_pos:], ring[:write_pos]])

                f, t, S_db = compute_spectrogram(audio, SAMPLERATE)
                if FREQ_MAX is not None:
                    f_mask = f <= FREQ_MAX
                    f_plot = f[f_mask]
                    S_plot = S_db[f_mask, :]
                else:
                    f_plot = f
                    S_plot = S_db

                # Update image
                im.set_data(S_plot)
                im.set_extent([-WINDOW_SECONDS, 0.0, f_plot[0], f_plot[-1]])

                # Optional: auto-level contrast (comment out if you prefer fixed)
                # vmax = float(np.percentile(S_plot, 99))
                # vmin = vmax - 70
                # im.set_clim(vmin=vmin, vmax=vmax)

                ax.set_ylim(f_plot[0], f_plot[-1])
                fig.canvas.draw_idle()

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf this is an input-device issue, here are available input devices:")
        for i, name in get_available_input_devices():
            print(f"  {i}: {name}")

if __name__ == "__main__":
    main()