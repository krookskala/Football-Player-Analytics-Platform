# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
from typing import Dict, List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


class ClusterVisualizer:
    """
    Publication-quality visualizations for cluster analysis.

    Attributes:
        clustered_data (pd.DataFrame): Data with cluster assignments
        scaled_features (pd.DataFrame): Scaled KPI features
        raw_kpi_cols (list): Raw KPI column names
        cluster_names (dict): Tactical names for clusters
        position_name (str): Position being analyzed
        output_dir (str): Directory for saving plots

    Example:
        >>> viz = ClusterVisualizer(clustered_data, scaled_features,
                                    raw_kpi_cols, cluster_names, 'Midfielder')
        >>> viz.plot_all('output_dir/')
    """

    def __init__(self,
                 clustered_data: pd.DataFrame,
                 scaled_features: pd.DataFrame,
                 raw_kpi_cols: List[str],
                 cluster_names: Dict,
                 position_name: str):
        """Initialize the cluster visualizer."""
        self.clustered_data = clustered_data
        self.scaled_features = scaled_features
        self.raw_kpi_cols = raw_kpi_cols
        self.cluster_names = cluster_names
        self.position_name = position_name
        self.n_clusters = clustered_data['cluster_label'].nunique()

        # Style settings
        plt.style.use('default')
        sns.set_palette("Set2")

        print(f"Initialized ClusterVisualizer for {position_name}")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  KPIs: {len(raw_kpi_cols)}")

    def plot_pca_scatter(self, save_path: str = None) -> None:
        """
        Plot PCA 2D scatter with cluster coloring.

        Reduces high-dimensional KPI space to 2D using PCA for visualization.
        Shows cluster separation and explained variance.

        Args:
            save_path: Path to save figure (optional)
        """
        print("\nGenerating PCA scatter plot...")

        X = self.scaled_features.values
        labels = self.clustered_data['cluster_label'].values

        # Fit PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each cluster
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_name = self.cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
            n_players = mask.sum()

            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=f'{cluster_name} (n={n_players})',
                      alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

        # Calculate and plot centroids
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            centroid = X_pca[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1],
                      marker='X', s=300, c='red',
                      edgecolors='black', linewidths=2, zorder=10)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                     fontsize=12, fontweight='bold')
        ax.set_title(f'PCA Cluster Visualization - {self.position_name}\n'
                    f'Total Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

            # Save PCA model
            model_path = save_path.replace('.png', '_pca_model.pkl')
            joblib.dump(pca, model_path)
            print(f"  Saved PCA model: {model_path}")

        plt.close()

    def plot_cluster_distribution(self, save_path: str = None) -> None:
        """
        Plot bar chart of cluster size distribution.

        Shows role diversity and cluster balance.

        Args:
            save_path: Path to save figure (optional)
        """
        print("\nGenerating cluster distribution bar chart...")

        cluster_counts = self.clustered_data['cluster_label'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(self.n_clusters), cluster_counts.values,
                     color=sns.color_palette("Set2", self.n_clusters),
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            height = bar.get_height()
            pct = (count / len(self.clustered_data)) * 100
            cluster_name = self.cluster_names.get(i, {}).get('name', f'Cluster {i}')
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., -2,
                   cluster_name,
                   ha='center', va='top', fontsize=9, rotation=0)

        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
        ax.set_title(f'Cluster Membership Distribution - {self.position_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.n_clusters))
        ax.set_xticklabels([f'{i}' for i in range(self.n_clusters)])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        plt.close()

    def plot_radar_charts(self, save_dir: str = None) -> None:
        """
        Generate radar chart for each cluster.

        Shows cluster profile compared to position average.

        Args:
            save_dir: Directory to save figures (optional)
        """
        print("\nGenerating radar charts...")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Calculate position averages
        position_avg = self.clustered_data[self.raw_kpi_cols].mean()

        for cluster_id in range(self.n_clusters):
            cluster_data = self.clustered_data[
                self.clustered_data['cluster_label'] == cluster_id
            ]
            cluster_mean = cluster_data[self.raw_kpi_cols].mean()
            cluster_name = self.cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
            n_players = len(cluster_data)

            # Normalize values to 0-1 scale for radar chart
            max_vals = self.clustered_data[self.raw_kpi_cols].max()
            min_vals = self.clustered_data[self.raw_kpi_cols].min()

            cluster_norm = (cluster_mean - min_vals) / (max_vals - min_vals + 0.001)
            position_norm = (position_avg - min_vals) / (max_vals - min_vals + 0.001)

            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # Number of variables
            num_vars = len(self.raw_kpi_cols)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            # Plot cluster profile
            values = cluster_norm.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=cluster_name, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')

            # Plot position average
            values_avg = position_norm.tolist()
            values_avg += values_avg[:1]
            ax.plot(angles, values_avg, 'o--', linewidth=2, label='Position Average',
                   color='gray', alpha=0.7)

            # Fix axis to go from 0 to 1
            ax.set_ylim(0, 1)

            # Labels
            labels = [kpi.replace('_per_90', '').replace('_pct', '%').replace('_', ' ').title()
                     for kpi in self.raw_kpi_cols]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=10)

            ax.set_title(f'{cluster_name} (n={n_players})\n{self.position_name}',
                        size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)

            plt.tight_layout()

            if save_dir:
                save_path = os.path.join(save_dir, f'{self.position_name.lower().replace(" ", "_")}_cluster_{cluster_id}_radar.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  Saved: {save_path}")

            plt.close()

    def plot_z_score_heatmap(self, z_scores: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot Z-score heatmap for cluster comparison.

        Args:
            z_scores: DataFrame with Z-scores (from cluster_profiling)
            save_path: Path to save figure (optional)
        """
        print("\nGenerating Z-score heatmap...")

        # Prepare data (transpose so KPIs are rows, clusters are columns)
        heatmap_data = z_scores.set_index('cluster_id')[self.raw_kpi_cols].T

        # Rename columns with cluster names
        col_names = [self.cluster_names.get(i, {}).get('name', f'Cluster {i}')
                    for i in range(self.n_clusters)]
        heatmap_data.columns = col_names

        # Rename rows (KPI labels)
        row_names = [kpi.replace('_per_90', '').replace('_pct', '%').replace('_', ' ').title()
                    for kpi in self.raw_kpi_cols]
        heatmap_data.index = row_names

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Z-Score'}, linewidths=1, linecolor='black',
                   vmin=-2, vmax=2, ax=ax)

        ax.set_title(f'Cluster KPI Comparison (Z-Scores) - {self.position_name}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('KPI', fontsize=12, fontweight='bold')

        # Add interpretation text
        fig.text(0.5, 0.02,
                'Interpretation: Z > 1 (Strong), Z < -1 (Weak), |Z| < 1 (Average)',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        plt.close()

    def plot_top_kpis_boxplot(self, f_statistics: Dict, save_path: str = None) -> None:
        """
        Plot box plots for top 3 discriminative KPIs.

        Args:
            f_statistics: Dictionary of F-statistics (from cluster_profiling)
            save_path: Path to save figure (optional)
        """
        print("\nGenerating box plots for top 3 discriminative KPIs...")

        # Get top 3 KPIs
        top_kpis = list(f_statistics.items())[:3]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (kpi, f_stat) in enumerate(top_kpis):
            ax = axes[idx]

            # Prepare data for box plot
            data_to_plot = []
            labels = []

            for cluster_id in range(self.n_clusters):
                cluster_data = self.clustered_data[
                    self.clustered_data['cluster_label'] == cluster_id
                ][kpi]
                data_to_plot.append(cluster_data.values)
                cluster_name = self.cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
                labels.append(f'{cluster_name}\n(n={len(cluster_data)})')

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))

            # Color boxes by cluster
            colors = sns.color_palette("Set2", self.n_clusters)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            kpi_label = kpi.replace('_per_90', ' /90').replace('_pct', ' %').replace('_', ' ').title()
            ax.set_title(f'{kpi_label}\n(F={f_stat:.1f})',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(kpi_label, fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=0, labelsize=9)

        fig.suptitle(f'Top 3 Discriminative KPIs - {self.position_name}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        plt.close()

    def plot_all(self,
                 output_base_dir: str,
                 z_scores: pd.DataFrame,
                 f_statistics: Dict) -> Dict[str, str]:
        """
        Generate all visualizations and save to output directory.

        Args:
            output_base_dir: Base output directory
            z_scores: Z-scores from cluster_profiling
            f_statistics: F-statistics from cluster_profiling

        Returns:
            Dictionary with paths to all saved plots
        """
        print("\n" + "="*70)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*70)

        pos_name_safe = self.position_name.lower().replace(' ', '_')

        # Create output directories
        pca_dir = os.path.join(output_base_dir, 'dimensionality_reduction')
        dist_dir = os.path.join(output_base_dir, 'distributions')
        profile_dir = os.path.join(output_base_dir, 'cluster_profiles')

        os.makedirs(pca_dir, exist_ok=True)
        os.makedirs(dist_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)

        paths = {}

        # 1. PCA scatter
        pca_path = os.path.join(pca_dir, f'{pos_name_safe}_pca_clusters.png')
        self.plot_pca_scatter(save_path=pca_path)
        paths['pca'] = pca_path

        # 2. Distribution bar chart
        dist_path = os.path.join(dist_dir, f'{pos_name_safe}_cluster_distribution.png')
        self.plot_cluster_distribution(save_path=dist_path)
        paths['distribution'] = dist_path

        # 3. Radar charts (one per cluster)
        self.plot_radar_charts(save_dir=profile_dir)
        paths['radar_charts'] = profile_dir

        # 4. Z-score heatmap
        heatmap_path = os.path.join(profile_dir, f'{pos_name_safe}_z_score_heatmap.png')
        self.plot_z_score_heatmap(z_scores, save_path=heatmap_path)
        paths['heatmap'] = heatmap_path

        # 5. Box plots
        boxplot_path = os.path.join(profile_dir, f'{pos_name_safe}_top_kpis_boxplot.png')
        self.plot_top_kpis_boxplot(f_statistics, save_path=boxplot_path)
        paths['boxplot'] = boxplot_path

        print("\n" + "="*70)
        print("[SUCCESS] All visualizations generated!")
        print("="*70)

        return paths


if __name__ == "__main__":
    # Test placeholder
    print("Cluster Visualization Module")
    print("Use via ClusterVisualizer class")
