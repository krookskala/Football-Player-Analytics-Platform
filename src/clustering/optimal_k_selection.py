import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from typing import Dict, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimalKSelector:
    """
    Multi-metric framework for optimal K selection in K-Means clustering.

    This class implements four validation metrics to provide robust,
    objective recommendations for the number of clusters.

    Attributes:
        data (np.ndarray): Scaled feature matrix (n_samples, n_features)
        k_range (tuple): Min and max k values to test (inclusive)
        results (dict): Storage for all computed metrics

    Example:
        >>> selector = OptimalKSelector(scaled_features, k_range=(2, 8))
        >>> recommendation = selector.find_optimal_k()
        >>> print(f"Optimal k: {recommendation['optimal_k']}")
    """

    def __init__(self, data: np.ndarray, k_range: Tuple[int, int] = (2, 8)):
        """
        Initialize the optimal K selector.

        Args:
            data: Scaled feature matrix (n_samples, n_features)
            k_range: Tuple (min_k, max_k) to test (inclusive)

        Raises:
            ValueError: If k_range is invalid or data has insufficient samples
        """
        self.data = data
        self.k_range = k_range
        self.results = {
            'k_values': [],
            'wcss': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }

        # Validation
        if k_range[0] < 2:
            raise ValueError("Minimum k must be at least 2")
        if k_range[1] > len(data):
            raise ValueError(f"Maximum k ({k_range[1]}) exceeds sample size ({len(data)})")
        if k_range[0] >= k_range[1]:
            raise ValueError("min_k must be less than max_k")

        logger.info(f"Initialized OptimalKSelector")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  K range: {k_range[0]} to {k_range[1]}")

    def compute_wcss(self) -> List[float]:
        """
        Compute Within-Cluster Sum of Squares (WCSS) for Elbow Method.

        WCSS measures cluster compactness. The "elbow" in the WCSS curve
        indicates a point of diminishing returns for additional clusters.

        Formula:
            WCSS = Σ(i=1 to k) Σ(xj ∈ Ci) ||xj - μi||²

        Returns:
            List of WCSS values for each k
        """
        logger.info("\nComputing WCSS (Elbow Method)...")
        wcss_values = []

        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(self.data)
            wcss_values.append(kmeans.inertia_)
            self.results['k_values'].append(k)
            logger.info(f"  k={k}: WCSS={kmeans.inertia_:.2f}")

        self.results['wcss'] = wcss_values
        return wcss_values

    def compute_silhouette(self) -> List[float]:
        """
        Compute Silhouette Scores for each k value.

        Silhouette score measures how similar an object is to its own cluster
        compared to other clusters. Range: [-1, 1], higher is better.

        Score interpretation:
            > 0.7: Strong structure
            0.5-0.7: Reasonable structure
            0.25-0.5: Weak structure
            < 0.25: No substantial structure

        Formula:
            s(i) = (b(i) - a(i)) / max(a(i), b(i))
            where a(i) = avg distance to same cluster
                  b(i) = avg distance to nearest different cluster

        Returns:
            List of silhouette scores for each k

        Reference:
            Rousseeuw, P.J. (1987). Silhouettes: A graphical aid to the
            interpretation and validation of cluster analysis.
        """
        logger.info("\nComputing Silhouette Scores...")
        silhouette_scores = []

        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)
            logger.info(f"  k={k}: Silhouette={score:.3f}")

        self.results['silhouette'] = silhouette_scores
        return silhouette_scores

    def compute_davies_bouldin(self) -> List[float]:
        """
        Compute Davies-Bouldin Index for each k value.

        Davies-Bouldin Index measures cluster separation quality.
        Lower values indicate better clustering (well-separated clusters).

        Formula:
            DB = (1/k) Σ max((σi + σj) / d(ci, cj))
            where σi = avg distance within cluster i
                  d(ci, cj) = distance between centroids i and j

        Returns:
            List of Davies-Bouldin scores for each k (lower is better)

        Reference:
            Davies, D.L. & Bouldin, D.W. (1979). A Cluster Separation Measure.
            IEEE Transactions on Pattern Analysis and Machine Intelligence.
        """
        logger.info("\nComputing Davies-Bouldin Index...")
        db_scores = []

        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
            labels = kmeans.fit_predict(self.data)
            score = davies_bouldin_score(self.data, labels)
            db_scores.append(score)
            logger.info(f"  k={k}: Davies-Bouldin={score:.3f} (lower is better)")

        self.results['davies_bouldin'] = db_scores
        return db_scores

    def compute_calinski_harabasz(self) -> List[float]:
        """
        Compute Calinski-Harabasz Score (Variance Ratio Criterion).

        Measures the ratio of between-cluster dispersion to within-cluster
        dispersion. Higher values indicate better-defined clusters.

        Formula:
            CH = (SSB / SSW) × ((n - k) / (k - 1))
            where SSB = between-cluster sum of squares
                  SSW = within-cluster sum of squares
                  n = number of samples
                  k = number of clusters

        Returns:
            List of Calinski-Harabasz scores for each k (higher is better)

        Reference:
            Calinski, T. & Harabasz, J. (1974). A dendrite method for
            cluster analysis. Communications in Statistics.
        """
        logger.info("\nComputing Calinski-Harabasz Score...")
        ch_scores = []

        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=42
            )
            labels = kmeans.fit_predict(self.data)
            score = calinski_harabasz_score(self.data, labels)
            ch_scores.append(score)
            logger.info(f"  k={k}: Calinski-Harabasz={score:.2f} (higher is better)")

        self.results['calinski_harabasz'] = ch_scores
        return ch_scores

    def find_optimal_k(self) -> Dict:
        """
        Determine optimal k using all four metrics and provide recommendation.

        This method computes all metrics and provides a data-driven
        recommendation. The final recommendation considers:
        1. Highest Silhouette score
        2. Lowest Davies-Bouldin score
        3. Highest Calinski-Harabasz score
        4. Elbow point in WCSS curve (heuristic)

        Returns:
            Dictionary containing:
                - optimal_k: Recommended number of clusters
                - confidence: 'high', 'medium', or 'low'
                - metrics: All computed metric values
                - reasoning: Textual explanation of recommendation

        Example:
            >>> result = selector.find_optimal_k()
            >>> print(result['optimal_k'])  # 4
            >>> print(result['reasoning'])
        """
        logger.info("\n" + "="*60)
        logger.info("OPTIMAL K SELECTION - Multi-Metric Analysis")
        logger.info("="*60)

        # Compute all metrics
        if not self.results['wcss']:
            self.compute_wcss()
        if not self.results['silhouette']:
            self.compute_silhouette()
        if not self.results['davies_bouldin']:
            self.compute_davies_bouldin()
        if not self.results['calinski_harabasz']:
            self.compute_calinski_harabasz()

        k_values = list(range(self.k_range[0], self.k_range[1] + 1))

        # Find optimal k for each metric
        optimal_silhouette_idx = np.argmax(self.results['silhouette'])
        optimal_db_idx = np.argmin(self.results['davies_bouldin'])
        optimal_ch_idx = np.argmax(self.results['calinski_harabasz'])

        optimal_k_silhouette = k_values[optimal_silhouette_idx]
        optimal_k_db = k_values[optimal_db_idx]
        optimal_k_ch = k_values[optimal_ch_idx]

        # Elbow detection (simple heuristic: max second derivative)
        wcss = np.array(self.results['wcss'])
        if len(wcss) >= 3:
            second_derivative = np.diff(wcss, n=2)
            elbow_idx = np.argmax(second_derivative) + 1  # +1 due to diff offset
            optimal_k_elbow = k_values[elbow_idx]
        else:
            optimal_k_elbow = k_values[0]

        logger.info("\n" + "-"*60)
        logger.info("METRIC-SPECIFIC RECOMMENDATIONS:")
        logger.info("-"*60)
        logger.info(f"Elbow Method:        k = {optimal_k_elbow}")
        logger.info(f"Silhouette Score:    k = {optimal_k_silhouette} (score={self.results['silhouette'][optimal_silhouette_idx]:.3f})")
        logger.info(f"Davies-Bouldin:      k = {optimal_k_db} (score={self.results['davies_bouldin'][optimal_db_idx]:.3f})")
        logger.info(f"Calinski-Harabasz:   k = {optimal_k_ch} (score={self.results['calinski_harabasz'][optimal_ch_idx]:.2f})")

        # Consensus-based recommendation
        recommendations = [optimal_k_silhouette, optimal_k_db, optimal_k_ch, optimal_k_elbow]
        recommendation_counts = {k: recommendations.count(k) for k in set(recommendations)}

        # Choose k with most "votes" from metrics
        if max(recommendation_counts.values()) >= 3:
            # Strong consensus (3+ metrics agree)
            optimal_k = max(recommendation_counts, key=recommendation_counts.get)
            confidence = 'high'
            reasoning = f"Strong consensus: {recommendation_counts[optimal_k]}/4 metrics recommend k={optimal_k}"
        elif max(recommendation_counts.values()) == 2:
            # Moderate consensus (2 metrics agree)
            # Prefer Silhouette if it's one of the agreeing metrics
            optimal_k = optimal_k_silhouette
            confidence = 'medium'
            reasoning = f"Moderate consensus. Selected k={optimal_k} based on highest Silhouette score"
        else:
            # No consensus - default to Silhouette (most interpretable)
            optimal_k = optimal_k_silhouette
            confidence = 'low'
            reasoning = f"No clear consensus. Selected k={optimal_k} based on Silhouette score (most reliable single metric)"

        logger.info("\n" + "="*60)
        logger.info(f"FINAL RECOMMENDATION: k = {optimal_k}")
        logger.info(f"Confidence: {confidence.upper()}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info("="*60)

        return {
            'optimal_k': optimal_k,
            'confidence': confidence,
            'reasoning': reasoning,
            'metrics': {
                'k_values': k_values,
                'wcss': self.results['wcss'],
                'silhouette': self.results['silhouette'],
                'davies_bouldin': self.results['davies_bouldin'],
                'calinski_harabasz': self.results['calinski_harabasz']
            },
            'metric_recommendations': {
                'elbow': optimal_k_elbow,
                'silhouette': optimal_k_silhouette,
                'davies_bouldin': optimal_k_db,
                'calinski_harabasz': optimal_k_ch
            }
        }

    def plot_metrics(self, save_path: str = None, position_name: str = ""):
        """
        Generate publication-quality plots for all four metrics.

        Creates a 2x2 subplot figure showing:
        1. Elbow plot (WCSS)
        2. Silhouette scores
        3. Davies-Bouldin index
        4. Calinski-Harabasz score

        Args:
            save_path: Path to save the figure (optional)
            position_name: Position name for title (e.g., "Midfielder")
        """
        if not self.results['k_values']:
            raise ValueError("No results to plot. Run find_optimal_k() first.")

        k_values = self.results['k_values']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Optimal K Selection - {position_name}', fontsize=16, fontweight='bold')

        # 1. Elbow Plot (WCSS)
        ax1 = axes[0, 0]
        ax1.plot(k_values, self.results['wcss'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)

        # 2. Silhouette Score
        ax2 = axes[0, 1]
        ax2.plot(k_values, self.results['silhouette'], 'go-', linewidth=2, markersize=8)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis (higher is better)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        ax2.legend()

        # Mark optimal k
        optimal_idx = np.argmax(self.results['silhouette'])
        ax2.plot(k_values[optimal_idx], self.results['silhouette'][optimal_idx],
                'r*', markersize=20, label=f'Optimal k={k_values[optimal_idx]}')

        # 3. Davies-Bouldin Index
        ax3 = axes[1, 0]
        ax3.plot(k_values, self.results['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax3.set_ylabel('Davies-Bouldin Index', fontsize=12)
        ax3.set_title('Davies-Bouldin Index (lower is better)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(k_values)

        # Mark optimal k
        optimal_idx = np.argmin(self.results['davies_bouldin'])
        ax3.plot(k_values[optimal_idx], self.results['davies_bouldin'][optimal_idx],
                'g*', markersize=20, label=f'Optimal k={k_values[optimal_idx]}')
        ax3.legend()

        # 4. Calinski-Harabasz Score
        ax4 = axes[1, 1]
        ax4.plot(k_values, self.results['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax4.set_ylabel('Calinski-Harabasz Score', fontsize=12)
        ax4.set_title('Calinski-Harabasz Score (higher is better)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(k_values)

        # Mark optimal k
        optimal_idx = np.argmax(self.results['calinski_harabasz'])
        ax4.plot(k_values[optimal_idx], self.results['calinski_harabasz'][optimal_idx],
                'r*', markersize=20, label=f'Optimal k={k_values[optimal_idx]}')
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"\nPlot saved to: {save_path}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Optimal K Selection Module - Test")
    logger.info("="*60)

    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    test_data = np.random.randn(n_samples, n_features)

    # Run optimal k selection
    selector = OptimalKSelector(test_data, k_range=(2, 8))
    result = selector.find_optimal_k()

    logger.info("\nTest completed successfully!")
