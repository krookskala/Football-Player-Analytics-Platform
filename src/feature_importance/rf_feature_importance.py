# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RFFeatureImportanceAnalyzer:
    """
    Random Forest-based feature importance analyzer for cluster validation.

    This class trains a Random Forest classifier to predict cluster labels
    and extracts feature importances to validate F-statistics from clustering.

    Parameters
    ----------
    position_name : str
        Position name (e.g., 'Midfielder', 'Forward')
    clustered_data_path : str
        Path to clustered CSV file
    f_statistics_path : str
        Path to F-statistics JSON file
    n_estimators : int, default=100
        Number of trees in Random Forest
    random_state : int, default=42
        Random seed for reproducibility
    variance_threshold : float, default=0.01
        Minimum variance for feature selection
    """

    def __init__(self, position_name, clustered_data_path, f_statistics_path,
                 n_estimators=100, random_state=42, variance_threshold=0.01):
        self.position_name = position_name
        self.clustered_data_path = clustered_data_path
        self.f_statistics_path = f_statistics_path
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.variance_threshold = variance_threshold

        # Data attributes
        self.clustered_data = None
        self.f_statistics = None
        self.X = None
        self.y = None
        self.feature_names = None

        # Model attributes
        self.rf_model = None
        self.feature_importances = None
        self.cv_scores = None
        self.accuracy = None

        # Comparison attributes
        self.importance_rankings = None
        self.spearman_rho = None
        self.spearman_pvalue = None

        logger.info(f"Initialized RFFeatureImportanceAnalyzer for {position_name}")

    def load_data(self):
        """Load clustered data and F-statistics."""
        logger.info(f"\nLoading data...")

        # Load clustered data
        self.clustered_data = pd.read_csv(self.clustered_data_path)
        logger.info(f"  Clustered data loaded: {self.clustered_data.shape}")

        # Load F-statistics
        with open(self.f_statistics_path, 'r', encoding='utf-8') as f:
            self.f_statistics = json.load(f)
        logger.info(f"  F-statistics loaded: {len(self.f_statistics)} features")

        # Remove NaN F-statistics (if any)
        self.f_statistics = {k: v for k, v in self.f_statistics.items()
                            if not np.isnan(v)}
        logger.info(f"  Valid F-statistics: {len(self.f_statistics)}")

    def prepare_features(self):
        """
        Prepare scaled features for Random Forest training.

        Uses scaled KPIs (e.g., 'ball_recoveries_per_90_scaled')
        and applies variance threshold filtering.
        """
        logger.info(f"\nPreparing features...")

        # Get scaled feature columns
        scaled_cols = [col for col in self.clustered_data.columns
                      if col.endswith('_scaled')]

        # Get raw KPI names (for matching with F-statistics)
        raw_kpi_names = [col.replace('_scaled', '') for col in scaled_cols]

        # Filter to only KPIs that have F-statistics
        valid_kpis = [kpi for kpi in raw_kpi_names
                     if kpi in self.f_statistics]
        valid_scaled_cols = [kpi + '_scaled' for kpi in valid_kpis]

        logger.info(f"  Available scaled features: {len(scaled_cols)}")
        logger.info(f"  Features with F-statistics: {len(valid_scaled_cols)}")

        # Extract features and target
        X = self.clustered_data[valid_scaled_cols].values
        y = self.clustered_data['cluster_label'].values

        # Variance threshold filtering
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_filtered = selector.fit_transform(X)

        # Get retained feature names
        retained_mask = selector.get_support()
        retained_features = [valid_kpis[i] for i, keep in enumerate(retained_mask) if keep]

        logger.info(f"  Features after variance filtering: {len(retained_features)}")

        # Store
        self.X = X_filtered
        self.y = y
        self.feature_names = retained_features

        logger.info(f"  Final feature matrix: {self.X.shape}")
        logger.info(f"  Target distribution: {np.bincount(self.y)}")

    def train_rf_classifier(self, n_folds=5):
        """
        Train Random Forest classifier with cross-validation.

        Parameters
        ----------
        n_folds : int, default=5
            Number of folds for StratifiedKFold CV
        """
        logger.info(f"\nTraining Random Forest Classifier...")
        logger.info(f"  n_estimators: {self.n_estimators}")
        logger.info(f"  n_folds: {n_folds}")
        logger.info(f"  random_state: {self.random_state}")

        # Initialize Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=self.random_state)
        self.cv_scores = cross_val_score(self.rf_model, self.X, self.y,
                                        cv=cv, scoring='accuracy')

        logger.info(f"  CV Accuracy Scores: {self.cv_scores}")
        logger.info(f"  Mean CV Accuracy: {self.cv_scores.mean():.3f} (+/- {self.cv_scores.std():.3f})")

        # Train final model on full data
        self.rf_model.fit(self.X, self.y)
        self.accuracy = self.rf_model.score(self.X, self.y)
        logger.info(f"  Training Accuracy: {self.accuracy:.3f}")

    def calculate_feature_importance(self):
        """
        Calculate and normalize feature importances.

        Returns Gini-based importances normalized to sum to 1.
        """
        logger.info(f"\nCalculating feature importances...")

        # Get raw importances
        raw_importances = self.rf_model.feature_importances_

        # Normalize (sum=1)
        self.feature_importances = raw_importances / raw_importances.sum()

        logger.info(f"  Feature importances calculated (normalized)")
        logger.info(f"  Sum of importances: {self.feature_importances.sum():.6f}")

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': self.feature_importances
        }).sort_values('rf_importance', ascending=False)

        logger.info(f"\nTop 5 Features (RF Importance):")
        for i, row in importance_df.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['rf_importance']:.4f}")

        return importance_df

    def compare_with_fstats(self):
        """
        Compare RF importance with F-statistics using Spearman correlation.

        Returns
        -------
        pd.DataFrame
            Comparison table with RF importance, F-statistics, and rankings
        """
        logger.info(f"\nComparing RF importance with F-statistics...")

        # Get feature importances
        importance_df = self.calculate_feature_importance()

        # Add F-statistics
        importance_df['f_statistic'] = importance_df['feature'].map(self.f_statistics)

        # Add rankings
        importance_df['rf_rank'] = importance_df['rf_importance'].rank(
            ascending=False, method='min'
        ).astype(int)
        importance_df['f_stat_rank'] = importance_df['f_statistic'].rank(
            ascending=False, method='min'
        ).astype(int)

        # Calculate Spearman correlation
        self.spearman_rho, self.spearman_pvalue = spearmanr(
            importance_df['rf_importance'],
            importance_df['f_statistic']
        )

        logger.info(f"  Spearman correlation (rho): {self.spearman_rho:.3f}")
        logger.info(f"  P-value: {self.spearman_pvalue:.4f}")

        # Interpretation
        if self.spearman_rho > 0.7:
            interpretation = "STRONG agreement"
        elif self.spearman_rho > 0.5:
            interpretation = "MODERATE agreement"
        else:
            interpretation = "WEAK agreement"
        logger.info(f"  Interpretation: {interpretation}")

        # Store
        self.importance_rankings = importance_df

        return importance_df

    def plot_importance(self, save_path, top_n=10):
        """
        Plot horizontal bar chart of feature importances.

        Parameters
        ----------
        save_path : str
            Path to save plot
        top_n : int, default=10
            Number of top features to plot
        """
        logger.info(f"\nGenerating importance barplot...")

        # Get top N features
        top_features = self.importance_rankings.head(top_n)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Horizontal barplot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['rf_importance'],
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('RF Importance Score (Normalized)', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.position_name} - Top {top_n} Features (Random Forest)',
                    fontsize=13, fontweight='bold', pad=15)

        # Add importance values as text
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['rf_importance'] + 0.005, i,
                   f"{row['rf_importance']:.3f}",
                   va='center', fontsize=9)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  [SAVED] {save_path}")

    def plot_fstat_vs_rf(self, save_path):
        """
        Plot scatter plot of F-statistics vs. RF importance.

        Parameters
        ----------
        save_path : str
            Path to save plot
        """
        logger.info(f"\nGenerating F-stat vs. RF scatter plot...")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # Scatter plot
        ax.scatter(self.importance_rankings['f_statistic'],
                  self.importance_rankings['rf_importance'],
                  s=100, alpha=0.6, color='coral', edgecolor='black', linewidth=0.5)

        # Add feature names
        for idx, row in self.importance_rankings.iterrows():
            ax.annotate(row['feature'],
                       (row['f_statistic'], row['rf_importance']),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')

        # Labels
        ax.set_xlabel('F-Statistic (Clustering)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RF Importance (Normalized)', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.position_name} - F-Statistic vs. RF Importance\n' +
                    f'Spearman rho = {self.spearman_rho:.3f} (p={self.spearman_pvalue:.4f})',
                    fontsize=13, fontweight='bold', pad=15)

        # Grid
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  [SAVED] {save_path}")

    def save_results(self, output_dir):
        """
        Save RF analysis results.

        Parameters
        ----------
        output_dir : str
            Output directory path

        Returns
        -------
        dict
            Paths to saved files
        """
        logger.info(f"\nSaving results to {output_dir}...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        position_safe = self.position_name.lower().replace(' ', '_')

        # 1. Save JSON (model metrics)
        json_path = output_dir / f'{position_safe}_rf_results.json'
        json_data = {
            'position': self.position_name,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'n_clusters': len(np.unique(self.y)),
            'model': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'max_depth': 10
            },
            'performance': {
                'cv_mean_accuracy': float(self.cv_scores.mean()),
                'cv_std_accuracy': float(self.cv_scores.std()),
                'cv_scores': self.cv_scores.tolist(),
                'training_accuracy': float(self.accuracy)
            },
            'validation': {
                'spearman_rho': float(self.spearman_rho),
                'spearman_pvalue': float(self.spearman_pvalue),
                'interpretation': 'STRONG' if self.spearman_rho > 0.7
                                 else 'MODERATE' if self.spearman_rho > 0.5
                                 else 'WEAK'
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  [SAVED] {json_path}")

        # 2. Save CSV (importance rankings)
        csv_path = output_dir / f'{position_safe}_importance_rankings.csv'
        self.importance_rankings.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"  [SAVED] {csv_path}")

        # 3. Save plots
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)

        importance_plot_path = plot_dir / f'{position_safe}_importance_barplot.png'
        self.plot_importance(importance_plot_path, top_n=10)

        scatter_plot_path = plot_dir / f'{position_safe}_fstat_vs_rf.png'
        self.plot_fstat_vs_rf(scatter_plot_path)

        return {
            'json': str(json_path),
            'csv': str(csv_path),
            'importance_plot': str(importance_plot_path),
            'scatter_plot': str(scatter_plot_path)
        }

    def run_full_analysis(self, output_dir):
        """
        Run complete RF feature importance analysis pipeline.

        Parameters
        ----------
        output_dir : str
            Output directory for results

        Returns
        -------
        dict
            Analysis summary
        """
        logger.info("="*70)
        logger.info(f"RF FEATURE IMPORTANCE ANALYSIS - {self.position_name.upper()}")
        logger.info("="*70)

        # 1. Load data
        self.load_data()

        # 2. Prepare features
        self.prepare_features()

        # 3. Train RF
        self.train_rf_classifier(n_folds=5)

        # 4. Compare with F-stats
        self.compare_with_fstats()

        # 5. Save results
        saved_paths = self.save_results(output_dir)

        # Summary
        logger.info("\n" + "="*70)
        logger.info(f"ANALYSIS COMPLETED - {self.position_name.upper()}")
        logger.info("="*70)
        logger.info(f"Samples: {len(self.y)}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Clusters: {len(np.unique(self.y))}")
        logger.info(f"CV Accuracy: {self.cv_scores.mean():.3f} (+/- {self.cv_scores.std():.3f})")
        logger.info(f"Spearman rho: {self.spearman_rho:.3f} (p={self.spearman_pvalue:.4f})")
        logger.info(f"\nTop 3 Features (RF):")
        for i, row in self.importance_rankings.head(3).iterrows():
            logger.info(f"  {row['rf_rank']}. {row['feature']}: " +
                 f"RF={row['rf_importance']:.3f}, F-stat={row['f_statistic']:.2f}")
        logger.info("="*70)

        return {
            'position': self.position_name,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names),
            'cv_accuracy': float(self.cv_scores.mean()),
            'spearman_rho': float(self.spearman_rho),
            'top_3_features': self.importance_rankings.head(3)['feature'].tolist(),
            'saved_paths': saved_paths
        }
