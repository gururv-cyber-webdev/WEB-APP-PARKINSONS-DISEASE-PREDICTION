# extract_script.py
import parselmouth
import random

def extract_jitter_shimmer(path, simulate_pd=False):
    snd = parselmouth.Sound(path)
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    def val(v):
        result = parselmouth.praat.call(
            pp if 'jitter' in v.lower() else [snd, pp],
            v, 0, 0, 0.0001, 0.02, 1.3, 1.6
        ) if 'shimmer' in v.lower() else parselmouth.praat.call(
            pp, v, 0, 0, 0.0001, 0.02, 1.3
        )
        return round(result + random.uniform(6.0, 10.0), 6) if simulate_pd else round(result, 6)

    features = {
        'Jitter(%)': val("Get jitter (local)"),
        'Jitter(Abs)': val("Get jitter (local, absolute)"),
        'Jitter(RAP)': val("Get jitter (rap)"),
        'Jitter(PPQ5)': val("Get jitter (ppq5)"),
        'Jitter:DDP': val("Get jitter (ddp)"),
        'Shimmer(local)': val("Get shimmer (local)"),
        'Shimmer(dB)': val("Get shimmer (local_dB)"),
        'Shimmer(APQ3)': val("Get shimmer (apq3)"),
        'Shimmer(APQ5)': val("Get shimmer (apq5)"),
        'Shimmer(APQ11)': val("Get shimmer (apq11)"),
        'Shimmer(DDA)': val("Get shimmer (dda)")
    }

    return features
