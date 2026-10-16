import time
import numpy as np


def DOA(x, fobj, lb, ub, T):
    pop, D = x.shape
    lb = np.ones(D) * lb if np.isscalar(lb) else np.array(lb)
    ub = np.ones(D) * ub if np.isscalar(ub) else np.array(ub)
    SELECT = np.arange(pop)
    sbest = np.ones(D)
    sbestd = np.ones((5, D))
    fbest = np.inf
    fbestd = np.ones(5) * np.inf
    fbest_history = np.zeros(T)

    ct = time.time()
    # Exploration Phase
    explore_iters = int(0.9 * T)
    for i in range(explore_iters):
        for m in range(5):
            start = int(m * pop / 5)
            end = int((m + 1) * pop / 5)
            k_low = max(1, int(np.ceil(D / (8 * (m + 1)))))
            k_high = max(k_low, int(np.ceil(D / (3 * (m + 1)))))
            k = np.random.randint(k_low, k_high + 1)
            # Update local best
            for j in range(start, end):
                fit = fobj(x[j])
                if fit < fbestd[m]:
                    fbestd[m] = fit
                    sbestd[m] = x[j].copy()

            # Memory strategy
            for j in range(start, end):
                x[j] = sbestd[m].copy()
                idx = np.random.permutation(D)[:k]
                if np.random.rand() < 0.9:
                    for h in idx:
                        step = (
                            np.random.rand() * (ub[h] - lb[h]) + lb[h]
                        ) * (np.cos((i + T / 10) * np.pi / T) + 1) / 2
                        x[j, h] += step
                        if x[j, h] > ub[h] or x[j, h] < lb[h]:
                            if D > 15:
                                candidates = np.delete(SELECT, j)
                                sel = np.random.choice(candidates)
                                x[j, h] = x[sel, h]
                            else:
                                x[j, h] = np.random.rand() * (ub[h] - lb[h]) + lb[h]

                else:
                    for h in idx:
                        sel = np.random.randint(pop)
                        x[j, h] = x[sel, h]
            if fbestd[m] < fbest:
                fbest = fbestd[m]
                sbest = sbestd[m].copy()
        fbest_history[i] = fbest

    # Exploitation Phase
    for i in range(explore_iters, T):
        for p in range(pop):
            fit = fobj(x[p])
            if fit < fbest:
                fbest = fit
                sbest = x[p].copy()
        for j in range(pop):
            km = max(2, int(np.ceil(D / 3)))
            k = np.random.randint(2, km + 1)
            x[j] = sbest.copy()
            idx = np.random.permutation(D)[:k]
            for h in idx:
                step = (
                    np.random.rand() * (ub[h] - lb[h]) + lb[h]
                ) * (np.cos(i * np.pi / T) + 1) / 2
                x[j, h] += step
                if x[j, h] > ub[h] or x[j, h] < lb[h]:
                    if D > 15:
                        candidates = np.delete(SELECT, j)
                        sel = np.random.choice(candidates)
                        x[j, h] = x[sel, h]
                    else:
                        x[j, h] = np.random.rand() * (
                            ub[h] - lb[h]
                        ) + lb[h]
        fbest_history[i] = fbest
    ct = time.time() - ct
    return fbest, fbest_history, sbest, ct