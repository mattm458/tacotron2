FEATURES_ALL = [
    "duration",
    "duration_vcd",
    "pitch_mean",
    "pitch_5",
    "pitch_95",
    "pitch_range",
    "pitch_mean_log",
    "pitch_5_log",
    "pitch_95_log",
    "pitch_range_log",
    "intensity_mean",
    "intensity_mean_vcd",
    "jitter",
    "shimmer",
    "nhr",
    "nhr_vcd",
    "rate",
    "rate_vcd",
]

FEATURES_ALL_SPEAKER_NORM = [f"{x}_speaker_norm" for x in FEATURES_ALL]
FEATURES_ALL_SPEAKER_NORM_CLIP = [f"{x}_clip" for x in FEATURES_ALL_SPEAKER_NORM]

FEATURES_ALL_DATASET_NORM = [f"{x}_dataset_norm" for x in FEATURES_ALL]
FEATURES_ALL_DATASET_NORM_CLIP = [f"{x}_clip" for x in FEATURES_ALL_DATASET_NORM]

FEATURES_ALL_DATASET_GENDER_NORM = [f"{x}_dataset_gender_norm" for x in FEATURES_ALL]
FEATURES_ALL_DATASET_GENDER_NORM_CLIP = [
    f"{x}_clip" for x in FEATURES_ALL_DATASET_GENDER_NORM
]

FEATURES_ALL_CV_NORM = [f"{x}_cv_norm" for x in FEATURES_ALL]
FEATURES_ALL_CV_NORM_CLIP = [f"{x}_clip" for x in FEATURES_ALL_CV_NORM]

FEATURES_ALL_CV_GENDER_NORM = [f"{x}_cv_norm" for x in FEATURES_ALL]
FEATURES_ALL_CV_GENDER_NORM_CLIP = [f"{x}_clip" for x in FEATURES_ALL_CV_GENDER_NORM]


def normalize(df, medians, stds):
    minimums = medians - (3 * stds)
    maximums = medians + (3 * stds)
    t_max, t_min = 1, -1

    return (((df - minimums) * (t_max - t_min)) / (maximums - minimums)) + t_min


def normalize_and_clip(df, feature_medians, feature_stds, F, F_CLIP):
    df[F] = normalize(df[FEATURES_ALL], medians=feature_medians, stds=feature_stds)
    df[F_CLIP] = df[F].clip(-1, 1)
