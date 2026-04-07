"""
K-Means Clustering Module for Football Player Performance Analysis

This module implements position-specific player archetype identification using
K-Means clustering with comprehensive validation metrics.

References:
    - Jain, A.K. (2010). Data clustering: 50 years beyond K-means.
      Pattern Recognition Letters, 31(8), 651-666.
    - Kaufman, L. & Rousseeuw, P.J. (2005). Finding Groups in Data:
      An Introduction to Cluster Analysis. Wiley.
    - Sarmento, H. et al. (2019). Player typology in elite football:
      A multidimensional approach. International Journal of Performance
      Analysis in Sport, 19(5), 800-820.

Modules:
    - optimal_k_selection: Multi-metric framework for determining optimal
      number of clusters (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz)
    - kmeans_clustering: K-Means implementation with robustness testing (ARI)
    - cluster_profiling: Statistical profiling and tactical naming
    - cluster_visualization: Publication-quality visualizations
"""

__version__ = "1.0.0"
__author__ = "Football Analytics Research"

# Module will be populated as we create each component
