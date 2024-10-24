#!/usr/bin/env python3.12

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

if __name__ == "__main__":
    # samples = np.linspace(0, 1, 1000)

    d1 = sp.stats.uniform.rvs(loc=0.7, scale=0.2, size=1000)
    d2 = sp.stats.uniform.rvs(loc=0.6, scale=0.4, size=1000)
    d3 = sp.stats.norm.rvs(loc=0.8, scale=0.1, size=1000)
    d4 = np.concat(
        [
            sp.stats.norm.rvs(loc=0.7, scale=0.05, size=500),
            sp.stats.norm.rvs(loc=0.9, scale=0.05, size=500),
        ]
    )

    # plt.hist(d3, bins=100, range=[0, 1])

    print(f"{d1.mean()=}")
    print(f"{d2.mean()=}")
    print(f"{d3.mean()=}")
    print(f"{d4.mean()=}")

    print(f"{d1.std()=}")
    print(f"{d2.std()=}")
    print(f"{d3.std()=}")
    print(f"{d4.std()=}")

    plt.boxplot([d1, d2, d3, d4])
    plt.violinplot([d1, d2, d3, d4])
    plt.show()
