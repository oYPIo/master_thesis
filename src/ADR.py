import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from tqdm import tqdm

def normalize_by_abs_sum(args):
    abs_sum = sum([abs(arg) for arg in args ])
    return [abs(arg)/abs_sum for arg in args ]

def compute_entropy(x, n_bins):
    """Estimate entropy using histogram bins."""
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_joint_mutual_info(X, y):
    """Compute joint mutual information for a set of bands with the label."""
    # Discretize and concatenate band pairs
    X_joint = np.array([hash(tuple(row)) for row in X])
    return mutual_info_score(X_joint, y)

def adaptive_band_selection(data, labels, num_bands=12, lam=0.3, n_bins=20, onlyScore=[], modified=False):
    H, W, B = data.shape
    X = data.reshape(-1, B)
    print(X.shape)
    y = labels.flatten()

    mask_labeled = y > 0
    Xl = X[mask_labeled]
    yl = y[mask_labeled] - 1  # zero-based
    Xu = X[~mask_labeled]

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    Xl_disc = discretizer.fit_transform(Xl)
    Xu_disc = discretizer.fit_transform(Xu)

    print(Xl[:6, 0])
    print(Xl_disc[:6, 0])

    # Compute MI(Xl_j; C)
    mi_class = mutual_info_classif(Xl_disc, yl, discrete_features=True)
    print('Mutual Information between bands and labels')
    for band in range(0, mi_class.shape[0], mi_class.shape[0]//4):
        print(f'band {band}:', mi_class[band])
    print()

    # Compute H(Xu_j)
    ent_unlabeled = np.array([compute_entropy(Xu_disc[:, j], n_bins) for j in range(B)])
    print('Unlabled bands entropy')
    for band in range(0, ent_unlabeled.shape[0], ent_unlabeled.shape[0]//4):
        print(f'band {band}:', ent_unlabeled[band])
    print()

    selected = []
    remaining = list(range(B))
    final_score = 0
    score_comp_ratio = []

    if not onlyScore:
        for _ in tqdm(range(num_bands), desc="Band selection"):
            best_score = -np.inf
            best_band = None
            for band in remaining:
                S = selected + [band]
                d = len(S)

                # Relevance
                rel = np.sum(mi_class[S]) / d

                # Discriminative redundancy: joint MI with class
                disred = 0
                if d > 1:
                    for j, m in combinations(S, 2):
                        disred += compute_joint_mutual_info(Xl_disc[:, [j, m]], yl)
                    disred = 2 * disred / (d **2 - d)

                # Diversity (entropy of each band)
                div = np.sum(ent_unlabeled[S]) / d

                # Redundancy among unlabeled bands
                redun = 0
                if d > 1:
                    for j, m in combinations(S, 2):
                        redun += mutual_info_score(Xu_disc[:, j], Xu_disc[:, m])
                    redun = 2 * redun / (d * (d - 1))

                if not modified:
                    score = rel - disred + lam * (div - redun)
                else:
                    score = rel - disred - lam * (div + redun)

                if score > best_score:
                    best_score = score
                    best_band = band

                    comp_ratio = normalize_by_abs_sum([rel, disred, lam * div, lam* redun])
                    score_comp_ratio.append(comp_ratio)

            final_score = best_score
            selected.append(best_band)
            remaining.remove(best_band)
    else:
        S = onlyScore
        d = len(S)

        # Relevance
        rel = np.sum(mi_class[S]) / d

        # Discriminative redundancy: joint MI with class
        disred = 0
        if d > 1:
            for j, m in combinations(S, 2):
                disred += compute_joint_mutual_info(Xl_disc[:, [j, m]], yl)
            disred = 2 * disred / (d **2 - d)

        # Diversity (entropy of each band)
        div = np.sum(ent_unlabeled[S]) / d

        # Redundancy among unlabeled bands
        redun = 0
        if d > 1:
            for j, m in combinations(S, 2):
                redun += mutual_info_score(Xu_disc[:, j], Xu_disc[:, m])
            redun = 2 * redun / (d * (d - 1))
    

        if not modified:
            final_score = rel - disred + lam * (div - redun)
        else:
            final_score = rel - disred - lam * (div + redun)
        
        comp_ratio = normalize_by_abs_sum([rel, disred, lam * div, lam* redun])
        score_comp_ratio.append(comp_ratio)

    
    print(f'Final score: {final_score}\n')
    score_comp_mean_perc = np.mean(score_comp_ratio, axis=0) * 100
    print('Score comps mean ratio (%)')
    print(f' rel: {score_comp_mean_perc[0]:.0f}')
    print(f' disred: {score_comp_mean_perc[1]:.0f}')
    print(f' lam * div: {score_comp_mean_perc[2]:0f}')
    print(f' lam * redun: {score_comp_mean_perc[3]:0f}')

    return selected
