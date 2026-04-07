# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score
)
import joblib
from typing import Dict, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PositionClusterer:
    """
    K-Means clustering for football player performance analysis with
    robustness validation.

    This class implements K-Means clustering with multiple random seeds
    to verify cluster stability using the Adjusted Rand Index (ARI).

    Attributes:
        position_name (str): Position name (e.g., 'Midfielder')
        n_clusters (int): Number of clusters
        scaled_features (pd.DataFrame): Scaled KPI features
        raw_features (pd.DataFrame): Original (unscaled) KPI values
        player_info (pd.DataFrame): Player metadata
        kmeans_model (KMeans): Fitted K-Means model
        cluster_labels (np.ndarray): Assigned cluster labels
        robustness_scores (dict): ARI scores from multi-seed testing

    Example:
        >>> clusterer = PositionClusterer('Midfielder', n_clusters=3,
                                          scaled_features=X_scaled,
                                          raw_features=X_raw,
                                          player_info=player_df)
        >>> clusterer.fit_with_robustness_test()
        >>> clusterer.save_results('output_dir/')
    """

    def __init__(self,
                 position_name: str,
                 n_clusters: int,
                 scaled_features: pd.DataFrame,
                 raw_features: pd.DataFrame,
                 player_info: pd.DataFrame):
        """
        Initialize the position-specific clusterer.

        Args:
            position_name: Position name (e.g., 'Midfielder')
            n_clusters: Optimal number of clusters (from optimal_k_selection)
            scaled_features: DataFrame with scaled KPI columns
            raw_features: DataFrame with raw (unscaled) KPI values
            player_info: DataFrame with player metadata (id, name, team, etc.)

        Raises:
            ValueError: If n_clusters is invalid or data shapes don't match
        """
        self.position_name = position_name
        self.n_clusters = n_clusters
        self.scaled_features = scaled_features
        self.raw_features = raw_features
        self.player_info = player_info

        # Results storage
        self.kmeans_model = None
        self.cluster_labels = None
        self.silhouette_scores_per_sample = None
        self.distances_to_centroid = None
        self.robustness_scores = {}

        # Validation
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if n_clusters > len(scaled_features):
            raise ValueError(f"n_clusters ({n_clusters}) exceeds sample size ({len(scaled_features)})")
        if len(scaled_features) != len(raw_features) or len(scaled_features) != len(player_info):
            raise ValueError("scaled_features, raw_features, and player_info must have same length")

        logger.info(f"Initialized PositionClusterer for {position_name}")
        logger.info(f"  Samples: {len(scaled_features)}")
        logger.info(f"  Features: {scaled_features.shape[1]}")
        logger.info(f"  Target clusters: {n_clusters}")

    def fit_kmeans(self, random_state: int = 42) -> KMeans:
        """
        Fit K-Means clustering with specified random state.

        Uses k-means++ initialization for better convergence and
        reproducibility.

        Parameters used:
            - init='k-means++': Smart initialization (Arthur & Vassilvitskii, 2007)
            - n_init=10: Number of initializations to try
            - max_iter=300: Maximum iterations per run
            - random_state: For reproducibility

        Args:
            random_state: Random seed for reproducibility

        Returns:
            Fitted KMeans model

        Reference:
            Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages
            of careful seeding.
        """
        logger.info(f"\nFitting K-Means (k={self.n_clusters}, random_state={random_state})...")

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state,
            verbose=0
        )

        X = self.scaled_features.values
        kmeans.fit(X)

        logger.info(f"  Converged in {kmeans.n_iter_} iterations")
        logger.info(f"  Final inertia (WCSS): {kmeans.inertia_:.2f}")

        return kmeans

    def compute_cluster_quality_metrics(self, labels: np.ndarray) -> Dict:
        """
        Compute cluster quality metrics for given labels.

        Metrics include:
        - Overall silhouette score
        - Per-sample silhouette scores
        - Distances to cluster centroids

        Args:
            labels: Cluster assignments

        Returns:
            Dictionary with quality metrics
        """
        X = self.scaled_features.values

        # Overall silhouette score
        overall_silhouette = silhouette_score(X, labels)

        # Per-sample silhouette scores
        sample_silhouettes = silhouette_samples(X, labels)

        # Distance to assigned centroid
        centroids = self.kmeans_model.cluster_centers_
        distances = np.zeros(len(X))
        for i, (sample, label) in enumerate(zip(X, labels)):
            distances[i] = np.linalg.norm(sample - centroids[label])

        return {
            'overall_silhouette': overall_silhouette,
            'sample_silhouettes': sample_silhouettes,
            'distances_to_centroid': distances
        }

    def robustness_test(self, random_states: List[int] = [42, 123, 456]) -> Dict:
        """
        Test cluster stability using multiple random seeds and ARI.

        Adjusted Rand Index (ARI) measures agreement between two clusterings,
        corrected for chance. Range: [-1, 1], where:
            - 1.0: Perfect agreement
            - 0.0: Random labeling
            - < 0: Less agreement than expected by chance

        For robust clustering, we expect ARI > 0.8 between different seeds.

        Formula (Hubert & Arabie, 1985):
            ARI = (RI - E[RI]) / (max(RI) - E[RI])
            where RI = Rand Index, E[RI] = expected RI

        Args:
            random_states: List of random seeds to test

        Returns:
            Dictionary with ARI scores and statistics

        Reference:
            Hubert, L. & Arabie, P. (1985). Comparing partitions.
            Journal of Classification, 2(1), 193-218.
        """
        logger.info("\n" + "="*60)
        logger.info("ROBUSTNESS TESTING - Multi-Seed Clustering")
        logger.info("="*60)
        logger.info(f"Testing {len(random_states)} different random initializations...")

        all_labels = {}

        # Fit K-Means with each random state
        for rs in random_states:
            kmeans = self.fit_kmeans(random_state=rs)
            labels = kmeans.predict(self.scaled_features.values)
            all_labels[rs] = labels

        # Compute pairwise ARI scores
        ari_scores = []
        pairs = []

        for i, rs1 in enumerate(random_states):
            for rs2 in random_states[i+1:]:
                ari = adjusted_rand_score(all_labels[rs1], all_labels[rs2])
                ari_scores.append(ari)
                pairs.append((rs1, rs2))
                logger.info(f"  ARI (seed {rs1} vs {rs2}): {ari:.4f}")

        # Statistics
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        min_ari = np.min(ari_scores)
        max_ari = np.max(ari_scores)

        logger.info("\n" + "-"*60)
        logger.info("ROBUSTNESS STATISTICS:")
        logger.info("-"*60)
        logger.info(f"Mean ARI:   {mean_ari:.4f}")
        logger.info(f"Std ARI:    {std_ari:.4f}")
        logger.info(f"Min ARI:    {min_ari:.4f}")
        logger.info(f"Max ARI:    {max_ari:.4f}")

        # Interpretation
        if mean_ari >= 0.9:
            stability = "EXCELLENT"
            interpretation = "Clusters are highly stable across seeds"
        elif mean_ari >= 0.8:
            stability = "GOOD"
            interpretation = "Clusters are reasonably stable"
        elif mean_ari >= 0.6:
            stability = "MODERATE"
            interpretation = "Clusters show moderate stability"
        else:
            stability = "WEAK"
            interpretation = "Clusters are unstable - consider different k"

        logger.info(f"\nStability Assessment: {stability}")
        logger.info(f"Interpretation: {interpretation}")
        logger.info("="*60)

        return {
            'random_states': random_states,
            'all_labels': all_labels,
            'ari_scores': ari_scores,
            'pairs': pairs,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'min_ari': min_ari,
            'max_ari': max_ari,
            'stability': stability,
            'interpretation': interpretation
        }

    def fit_with_robustness_test(self, test_random_states: List[int] = [42, 123, 456]) -> None:
        """
        Fit K-Means and perform robustness testing.

        This is the main method to call. It:
        1. Performs robustness testing with multiple seeds
        2. Selects the seed with best silhouette score as final model
        3. Computes quality metrics for final model

        Args:
            test_random_states: Random seeds for robustness testing
        """
        # Perform robustness test
        robustness_results = self.robustness_test(random_states=test_random_states)
        self.robustness_scores = robustness_results

        # Choose best model (highest silhouette score)
        logger.info("\nSelecting final model...")
        best_silhouette = -1
        best_rs = test_random_states[0]

        for rs in test_random_states:
            labels = robustness_results['all_labels'][rs]
            sil_score = silhouette_score(self.scaled_features.values, labels)
            logger.info(f"  Seed {rs}: Silhouette = {sil_score:.3f}")

            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_rs = rs

        logger.info(f"\nSelected seed {best_rs} (Silhouette={best_silhouette:.3f})")

        # Fit final model with best seed
        self.kmeans_model = self.fit_kmeans(random_state=best_rs)
        self.cluster_labels = self.kmeans_model.predict(self.scaled_features.values)

        # Compute quality metrics
        quality = self.compute_cluster_quality_metrics(self.cluster_labels)
        self.silhouette_scores_per_sample = quality['sample_silhouettes']
        self.distances_to_centroid = quality['distances_to_centroid']

        logger.info(f"\nFinal Model Statistics:")
        logger.info(f"  Overall Silhouette Score: {quality['overall_silhouette']:.3f}")
        logger.info(f"  Mean ARI (robustness): {self.robustness_scores['mean_ari']:.3f}")

        # Cluster size distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        logger.info(f"\nCluster Size Distribution:")
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(self.cluster_labels)) * 100
            logger.info(f"  Cluster {cluster_id}: {count} players ({pct:.1f}%)")

    def get_clustered_data(self) -> pd.DataFrame:
        """
        Create final dataframe with all player data and cluster assignments.

        Returns:
            DataFrame with player info, raw KPIs, scaled KPIs, cluster labels,
            silhouette scores, and distances to centroids
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet. Call fit_with_robustness_test() first.")

        result = pd.concat([
            self.player_info.reset_index(drop=True),
            self.raw_features.reset_index(drop=True),
            self.scaled_features.reset_index(drop=True)
        ], axis=1)

        result['cluster_label'] = self.cluster_labels
        result['silhouette_score'] = self.silhouette_scores_per_sample
        result['distance_to_centroid'] = self.distances_to_centroid

        return result

    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save clustering results to files.

        Saves:
        1. KMeans model object (.pkl)
        2. Clustered player data (.csv)
        3. Robustness test results (.json)

        Args:
            output_dir: Directory to save files (will be created if not exists)

        Returns:
            Dictionary with paths to saved files
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Sanitize position name for filenames
        pos_name_safe = self.position_name.lower().replace(' ', '_')

        # 1. Save K-Means model
        model_path = os.path.join(output_dir, f'{pos_name_safe}_kmeans_model.pkl')
        joblib.dump(self.kmeans_model, model_path)

        # 2. Save clustered data
        data_path = os.path.join(output_dir, f'{pos_name_safe}_clustered.csv')
        clustered_data = self.get_clustered_data()
        clustered_data.to_csv(data_path, index=False, encoding='utf-8')

        # 3. Save robustness results
        robustness_path = os.path.join(output_dir, f'{pos_name_safe}_robustness.json')
        robustness_save = {
            'position': self.position_name,
            'n_clusters': self.n_clusters,
            'n_samples': len(self.scaled_features),
            'robustness': {
                'random_states': self.robustness_scores['random_states'],
                'ari_scores': self.robustness_scores['ari_scores'],
                'mean_ari': self.robustness_scores['mean_ari'],
                'std_ari': self.robustness_scores['std_ari'],
                'min_ari': self.robustness_scores['min_ari'],
                'max_ari': self.robustness_scores['max_ari'],
                'stability': self.robustness_scores['stability'],
                'interpretation': self.robustness_scores['interpretation']
            }
        }

        with open(robustness_path, 'w', encoding='utf-8') as f:
            json.dump(robustness_save, f, indent=2, ensure_ascii=False)

        logger.info(f"\n[SUCCESS] Results saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Data:  {data_path}")
        logger.info(f"  Robustness: {robustness_path}")

        return {
            'model': model_path,
            'data': data_path,
            'robustness': robustness_path
        }


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("K-Means Clustering Module - Test")
    logger.info("="*60)

    np.random.seed(42)
    n_samples = 100
    n_features = 8

    # Generate synthetic data
    X_scaled = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}_scaled' for i in range(n_features)]
    )

    X_raw = pd.DataFrame(
        np.random.rand(n_samples, n_features) * 10,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    player_info = pd.DataFrame({
        'player_id': range(n_samples),
        'player_name': [f'Player_{i}' for i in range(n_samples)],
        'team': ['Team_A'] * n_samples,
        'thesis_position': ['Test'] * n_samples
    })

    # Run clustering
    clusterer = PositionClusterer(
        position_name='Test Position',
        n_clusters=3,
        scaled_features=X_scaled,
        raw_features=X_raw,
        player_info=player_info
    )

    clusterer.fit_with_robustness_test()

    logger.info("\nTest completed successfully!")
