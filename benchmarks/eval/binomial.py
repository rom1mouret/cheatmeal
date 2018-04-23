#!/usr/bin/env python3

import math
import numpy as np


def null_hypothesis_probability(baseline_scores, higher_scores, shifts):
    # pre-compute binomial table with p=0.5 (null hypothesis)
    n = min(len(higher_scores), len(baseline_scores))
    n_factorial = math.factorial(n)
    bino = [n_factorial / (math.factorial(k) * math.factorial(n - k)) for k in range(n+1)]
    table = 1 - (np.cumsum(bino) - np.array(bino)) * 0.5 ** n

    for t in enumerate(table):
        print("P[n_success >= %i] = %f" % t)

    # number of times the 'higher' scores are above the shifted baseline
    baseline_scores = np.array(baseline_scores)[:n]
    higher_scores = np.array(higher_scores)[:n]
    print("baseline avg", baseline_scores.mean())
    print("higher scores avg", higher_scores.mean())

    successes = []
    for shift in shifts:
        shifted = baseline_scores + shift
        s = len(np.where(higher_scores > shifted)[0])
        successes.append(s)

    print("successes", successes, "out of n =", n)

    # probability of null hypothesis for different shift values
    prob = table[successes]
    return prob


def draw_graph(shifts, probabilities):
    import matplotlib.pyplot as plt

    plt.plot(shifts, probabilities)
    plt.title('probability that tested AE is that often better than baseline\n' + \
              'assuming null hypothesis (detectors both as likely to outperform the other)')
    plt.xlabel('mininimum margin x between paired measurements to count as "better"')
    plt.ylabel('probability')

    #plt.show()
    plt.savefig("kdd_signed_test.png", dpi=100)


if __name__ == "__main__":
    import sys
    import re

    fname, key1, key2 = sys.argv[1:]

    percent = re.compile("([0-9.]+)%")

    baseline = []
    better = []
    with open(fname, "r") as f:
        for line in f:
            m = percent.search(line)
            if m is not None:
                v = float(m.group(1))
                if key1 in line:
                    baseline.append(v)
                elif key2 in line:
                    better.append(v)

    print("baseline", baseline)
    print("allegedly better", better)

    #baseline = [73.2673, 71.756, 69.1803, 81.2217, 50.93715, 73.2142, 73.13109, 80.57471, 69.9167, 73.7083, 71.0884]
    #better = [71.9471, 70.8381, 74.75, 71.7194, 79.713, 80.4687, 81.25677, 80.574, 68.0142, 73.2491, 79.931]

    shifts = np.array([0, 0.5, 1.5, 2, 3, 4, 6, 8])
    #shifts = np.arange(0, 50, 10)
    probs = null_hypothesis_probability(baseline, better, shifts)
    draw_graph(shifts, probs)
