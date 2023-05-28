def normalize(df, medians, stds):
    minimums = medians - (3 * stds)
    maximums = medians + (3 * stds)
    t_max, t_min = 1, -1

    return (((df - minimums) * (t_max - t_min)) / (maximums - minimums)) + t_min
