import re
from collections import namedtuple

import hyphenate
import parselmouth
from nltk.corpus import cmudict

Timestamp = namedtuple("Timestamp", ["start", "end", "speaker", "text"])

MIN_DUR = 6.4 / 75


def ipus_to_turns(timestamps):
    turn_start = None
    turn_end = None
    turn_speaker = None
    turn_text = ""

    turns = []

    for start, end, speaker, text in timestamps:
        if turn_speaker is None:
            turn_speaker = speaker
            turn_start = start
            turn_end = end
            turn_text = text

        if turn_speaker == speaker:
            turn_end = end
            turn_text += " " + text
        else:
            turns.append(Timestamp(turn_start, turn_end, turn_speaker, turn_text))
            turn_start = start
            turn_end = end
            turn_speaker = speaker
            turn_text = text

    return turns


d = cmudict.dict()


def nsyl(word):
    """Count the number of syllables in a word."""
    # Remove trailing whitespace and convert to lowercase for dictionary lookup.
    word = word.strip().lower()

    # Special case: Empty string.
    if len(word) == 0:
        return 0

    if word == "((" or word == "))":
        return 0

    if word == "[noise]" or word == "[laughter]":
        return 0

    # Special case: Words consisting only of "?"
    # This situation occurs when the utterance is unclear, and annotators leave
    # one question mark per syllable.
    #
    # Note that, in the B-MIC, annotators do not leave punctuation except for
    # this case.
    num_qs = sum([1 if x == "?" else 0 for x in word])
    if num_qs == len(word):
        return num_qs

    # Special case: If the word ends in "-", remove the dash and try to determine
    # syllables as best we can. This occurs if the speaker does not complete the word.
    if word[-1] == "-":
        word = word[:-1]

    # Special case: If there is an apostrophe in the word, then it may not be
    # in the dictionary.
    if "'" in word:
        # A common situation is "'s", where the dictionary does not contain the possessive
        # form of all words. If that applies here, remove the "'s" and look up the
        # singular form of the word.
        if word not in d and word[-2:] == "'s":
            word = word[:-2]

    # Special case: Neither the dictionary or automatic hyphenation know what an
    # M&M is. If possessive ("m&m's"), it was made singular above.
    if word == "m&m" or word == "m&ms":
        return 3

    # Main syllable lookup functionality.
    if word in d:
        # If the word is in the dictionary, extract the syllable count.
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        # Otherwise, fall back to the hyphenate library for a best (but
        # sometimes inaccurate) guess.
        return len(hyphenate.hyphenate_word(word))


# def get_jitter_and_shimmer(sound, pitch, pulses):
#     if pitch is None or pulses is None:
#         return None, None

#     mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
#     mean_period = 1.0 / mean_pitch
#     num_voiced_frames = parselmouth.praat.call(pitch, "Count voiced frames")

#     if num_voiced_frames <= 0:
#         return None, None

#     textgrid = parselmouth.praat.call(pulses, "To TextGrid (vuv)", 0.02, mean_period)
#     intervals = parselmouth.praat.call(
#         [sound, textgrid], "Extract intervals", 1, "no", "V"
#     )

#     if type(intervals) is not list:
#         intervals = [intervals]

#     concatted = parselmouth.Sound.concatenate(intervals)

#     if concatted.get_total_duration() <= MIN_DUR:
#         return None, None

#     concatted_pitch = concatted.to_pitch()
#     concatted_pulses = parselmouth.praat.call([concatted_pitch], "To PointProcess")

#     jitter = parselmouth.praat.call(
#         concatted_pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
#     )

#     shimmer = parselmouth.praat.call(
#         [concatted, concatted_pulses],
#         "Get shimmer (local)",
#         0.0,
#         0.0,
#         0.0001,
#         0.02,
#         1.3,
#         1.6,
#     )

#     return jitter, shimmer


# nhr_re = re.compile("Mean noise-to-harmonics ratio: (\d*\.?\d+)")
# jitter_re = re.compile("Jitter \(local\): (\d*\.?\d+)%")
# shimmer_re = re.compile("Shimmer \(local\): (\d*\.?\d+)%")


# def get_features(sound):
#     try:
#         pitch = sound.to_pitch()
#         pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
#     except:
#         pitch = None
#         pulses = None

#     jitter, shimmer = get_jitter_and_shimmer(sound, pitch, pulses)

#     intensity_mean = intensity_std = intensity_min = intensity_max = energy = None
#     if sound.get_total_duration() > MIN_DUR:
#         intensity = sound.to_intensity()

#         intensity_mean = parselmouth.praat.call(
#             intensity, "Get mean", 0.0, 0.0, "Energy"
#         )
#         intensity_max = parselmouth.praat.call(
#             intensity, "Get maximum", 0.0, 0.0, "Parabolic"
#         )

#         intensity_min = parselmouth.praat.call(
#             intensity, "Get minimum", 0.0, 0.0, "Parabolic"
#         )

#     nhr = None

#     if sound and pitch and pulses:
#         report = str(
#             parselmouth.praat.call(
#                 [sound, pitch, pulses],
#                 "Voice report",
#                 0,
#                 0,
#                 75,
#                 500,
#                 1.3,
#                 1.6,
#                 0.03,
#                 0.45,
#             )
#         )

#         nhr = nhr_re.search(report)

#         if nhr:
#             nhr = nhr.group(1)

#     pitch_mean, pitch_max, pitch_min, pitch_range = None, None, None, None

#     if pitch is not None:
#         pitch_mean = (
#             parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "hertz")
#             if pitch
#             else None
#         )

#         pitch_95 = parselmouth.praat.call(
#             pitch, "Get quantile", 0.0, 0.0, 0.95, "hertz"
#         )
#         pitch_05 = parselmouth.praat.call(
#             pitch, "Get quantile", 0.0, 0.0, 0.05, "hertz"
#         )
#         pitch_range = pitch_95 - pitch_05
#     else:
#         return None

#     if nhr is None:
#         return None

#     return {
#         "pitch_mean": pitch_mean,
#         "pitch_range": pitch_range,
#         "intensity_mean": intensity_mean,
#         "jitter": jitter,
#         "shimmer": shimmer,
#         "nhr": float(nhr),
#         "duration": sound.duration,
#     }

with open("extract_features.praat") as infile:
    script = infile.read()


def get_features(sound):
    try:
        _, output = parselmouth.praat.run(sound, script, capture_output=True)

        features_in = [tuple(x.split(",")) for x in output.split("\n") if x]
        features_in = dict([(k, float(v)) for k, v in features_in])
        return features_in
    except:
        return None