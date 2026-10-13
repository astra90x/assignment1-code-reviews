"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# AB = [[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]]
#    = [[3,2],[2,3]].
# BA = [[1*3+2*2, 1*4+2*1], [0*3-1*2, 0*4-1*1]]
#    = [[7,6],[-2,-1]].
# A+B = [[4,6],[2,0]], so (A+B).T = [[4,2],[6,0]].
# AB.T = [[3*1+4*2, 3*0-4], [2*1+1*2, 2*0-1]] = [[11,-4],[4,-1]].
# trace(A) = 3+1 = 4; trace(AB) = 3+3 = 6.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# det(M-lambda*I) = (2-lambda)**2-1 = (lambda-1)(lambda-3).
# lambda=1: x-y=0, so choose v=(1,1); Mv=(1,1)=1*v.
# lambda=3: -x-y=0, so choose v=(1,-1); Mv=(3,-3)=3*v.
# Dividing these vectors by sqrt(2) gives an orthonormal basis.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# For A of shape (m,n) and B of shape (n,m),
# trace(AB) = sum_i sum_j A[i,j]*B[j,i]
#           = sum_j sum_i B[j,i]*A[i,j] = trace(BA).
# The sums are finite, and scalar multiplication commutes.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# det(C) = 4*6-7*2 = 10 != 0, so C is invertible.
# The 2x2 inverse formula gives C^-1 = (1/10)*[[6,-7],[-2,4]].
# C @ [[6,-7],[-2,4]] = [[24-14,-28+28],[12-12,-14+24]] = 10*I.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# 1 = c*sum_{k>=0} 3**k/k! = c*exp(3), hence c=exp(-3).
# E[X] = exp(-3)*sum_{k>=1} 3**k/(k-1)! = 3.
# E[X(X-1)] = exp(-3)*sum_{k>=2} 3**k/(k-2)! = 9.
# E[X**2] = 9+3 = 12, so Var(X) = 12-3**2 = 3.
# Replacing 3 by rate gives mean=variance=rate.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    return np.exp(-rate), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# P(A intersect B) = P(A|B)*P(B) = .6*.3 = .18.
# P(B|A) = .18/.5 = .36.
# P(A union B) = P(A)+P(B)-P(A intersect B) = .5+.3-.18 = .62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# Assume independent flips with a common theta. For this ordered sequence,
# L(theta) = theta**3*(1-theta)**2.
# log L = 3*log(theta)+2*log(1-theta).
# d(log L)/dtheta = 3/theta-2/(1-theta)=0 gives theta_MLE=3/5.
# The second derivative is -3/theta**2-2/(1-theta)**2 < 0.
# Bayes' rule gives posterior proportional to 2*theta**4*(1-theta)**2.
# Its log derivative is 4/theta-2/(1-theta)=0, so theta_MAP=2/3.
# The log second derivative is negative, and both endpoint densities vanish.
# In general MLE=h/n and MAP=(h+1)/(n+1). For h=0 the MLE is 0;
# for h=n both estimates are 1. These are boundary maxima.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# For independent Gaussian observations, write v=sigma**2>0:
# log L = -n/2*log(2*pi*v) - sum_i (x_i-mu)**2/(2*v).
# d(log L)/dmu = sum_i (x_i-mu)/v=0 gives mu_hat=mean(x)=4.
# d(log L)/dv = -n/(2*v)+sum_i (x_i-mu)**2/(2*v**2)=0.
# Thus v_hat=sum_i (x_i-mu_hat)**2/n = (9+1+1+9)/4=5.
# For each v, mean(x) uniquely maximizes the likelihood in mu.
# Substituting it, the v derivative has sign sum_i(x_i-mean(x))**2-n*v,
# so the likelihood increases up to v_hat and decreases afterward.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    mean = data.mean()
    return mean, np.mean((data - mean) ** 2)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# log p(mu|5) = constant - (5-mu)**2/8 - mu**2/2.
# Derivative: (5-mu)/4-mu=0, so mu_MAP=1.
# Second derivative: -1/4-1 < 0, proving a unique maximum.
# In general the mode is the precision-weighted average of x and prior_mean.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    precision = 1 / observation_variance + 1 / prior_variance
    return (x / observation_variance + prior_mean / prior_variance) / precision


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Let m=E[X], assuming E[X**2] is finite. By linearity of expectation,
# Var(X) = E[(X-m)**2] = E[X**2]-2*m*E[X]+m**2
#        = E[X**2]-m**2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Let mu=E[X] and Sigma=E[(X-mu)(X-mu).T]. Entry (i,j) is
# E[(X_i-mu_i)(X_j-mu_j)]. For fixed u, E[u.T X]=u.T mu, so
# Var(u.T X) = E[(u.T(X-mu))**2]
#            = E[u.T(X-mu)(X-mu).T u]
#            = u.T E[(X-mu)(X-mu).T] u = u.T Sigma u.
# The identity holds for any fixed u; u.T u=1 fixes the scale for PCA.
# Maximizing u.T Sigma u with L=u.T Sigma u-lambda*(u.T u-1)
# gives 2*Sigma*u-2*lambda*u=0. The variance at an eigenvector is lambda,
# so choose the largest eigenvalue, then the next orthogonal direction.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# Generate 500 samples with seed 0, mean [2,-1], covariance [[3,1.2],[1.2,1]].
# Population correlation is 1.2/sqrt(3*1)=0.6928; the sample value is 0.6992.
# The scatter slopes upward, with more spread in feature 1 than feature 2.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.asarray([2, -1] if mean is None else mean, dtype=float)
    covariance = np.asarray(
        [[3, 1.2], [1.2, 1]] if covariance is None else covariance, dtype=float
    )
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Rows are observations. Centering supplies X-mu in Q1. Then
# S=centered.T@centered/(n-1) estimates Sigma, and u.T S u is the
# sample variance of the projection. Symmetric eigh gives orthonormal axes.
# Sorting by decreasing eigenvalue puts the greatest-variance axis first.
# scores[:,0] alone is the one-dimensional representation.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (X.shape[0] - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # A covariance is positive semidefinite; remove negative roundoff at zero.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    scores = centered @ components
    return {
        'mean': mean,
        'centered': centered,
        'covariance': covariance,
        'eigenvalues': eigenvalues,
        'components': components,
        'scores': scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# Fraction j = lambda_j/sum_k(lambda_k), since sum(lambda)=trace(S).
# For this sample PC1 retains about 89.39% and PC2 about 10.61%.
# Keeping PC1 replaces two coordinates by one but loses the perpendicular
# spread. Both components retain all variance; a 95% target needs both.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# Equal axis scales preserve the angles between the principal directions.
# Direction segments pass through the sample mean and extend two standard
# deviations each way. The bar chart reports variance fractions, not accuracy.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, color='steelblue', label='Samples')
    for j, color in enumerate(['darkorange', 'seagreen']):
        offset = 2 * np.sqrt(result['eigenvalues'][j]) * result['components'][:, j]
        ends = np.array([result['mean'] - offset, result['mean'] + offset])
        ax.plot(ends[:, 0], ends[:, 1], color=color, linewidth=2.5, label=f'PC{j + 1}')
    ax.scatter(*result['mean'], color='black', marker='x', s=45, label='Sample mean')
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='Gaussian samples and principal directions')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    if result['eigenvalues'].sum() > 0:
        percentages = 100 * explained_variance(result['eigenvalues'])
        bars = ax.bar(['PC1', 'PC2'], percentages, color=['darkorange', 'seagreen'])
        ax.bar_label(bars, labels=[f'{value:.2f}%' for value in percentages], padding=4)
    else:
        ax.text(0.5, 0.5, 'Zero total variance: fractions undefined',
                ha='center', transform=ax.transAxes, fontsize=9)
        ax.set_xticks([0, 1], ['PC1', 'PC2'])
    ax.set(ylim=(0, 100), xlabel='Principal component',
           ylabel='Variance explained (%)', title='Share of total sample variance')
    ax.grid(axis='y', alpha=0.2)
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# ChatGPT/Codex drafted the derivations, code, plots, and numerical checks.
# Resources: the assignment README and starter, Lecture1_Intro.pdf, and
# Lecture2_Reviews.pdf. No external solution sources were used.
# The main distinction was the n divisor for Gaussian variance MLE versus
# n-1 for the PCA sample covariance. Both follow the lecture's definitions.

if __name__ == '__main__':
    # This block does not run when the grader imports your functions.
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=6, suppress=True)
    for name, value in zip(
        ['AB', 'BA', '(A+B).T', 'AB.T', 'trace(A)', 'trace(AB)'],
        matrix_operations(),
    ):
        print(f'{name}:\n{value}')
    print('M eigenvalues and eigenvectors:', eigensystem())
    print('C inverse:\n', matrix_inverse())
    print('Distribution (c, mean, variance):', distribution_moments())
    print('Event probabilities:', event_probabilities())
    print('Coin (MLE, MAP):', coin_estimates())
    print('Gaussian (mean MLE, variance MLE):', gaussian_mle())
    print('Gaussian mean MAP:', gaussian_map())

    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result['eigenvalues'])
    reconstruction = result['mean'] + result['scores'][:, :1] @ result['components'][:, :1].T
    print('Sample mean:', result['mean'])
    print('Sample covariance:\n', result['covariance'])
    print('Sample correlation:', np.corrcoef(X, rowvar=False)[0, 1])
    print('Eigenvalues:', result['eigenvalues'])
    print('Principal components (columns):\n', result['components'])
    print('Explained variance fractions:', fractions)
    print('Mean squared reconstruction distance:', np.mean(np.sum((X - reconstruction) ** 2, axis=1)))
    figure = make_plots(X, result)
    output = Path(__file__).resolve().parent / 'pca.png'
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f'Saved {output.name}')
