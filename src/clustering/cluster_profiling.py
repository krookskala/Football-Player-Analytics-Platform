# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from typing import Dict, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClusterProfiler:
    """
    Statistical profiling and tactical naming for player clusters.

    This class analyzes cluster characteristics, identifies discriminative
    KPIs, and assigns tactical names based on football domain knowledge.

    Attributes:
        clustered_data (pd.DataFrame): Data with cluster assignments
        raw_kpi_cols (list): Names of raw (unscaled) KPI columns
        position_name (str): Position being analyzed
        n_clusters (int): Number of clusters
        profiles (pd.DataFrame): Cluster profiles (mean values)
        f_statistics (dict): F-statistic for each KPI
        z_scores (pd.DataFrame): Z-scored cluster profiles

    Example:
        >>> profiler = ClusterProfiler(clustered_data, raw_kpi_cols, 'Midfielder')
        >>> profiler.generate_profiles()
        >>> profiler.calculate_f_statistics()
        >>> names = profiler.assign_tactical_names()
    """

    def __init__(self,
                 clustered_data: pd.DataFrame,
                 raw_kpi_cols: List[str],
                 position_name: str):
        """
        Initialize the cluster profiler.

        Args:
            clustered_data: DataFrame with cluster_label column and KPIs
            raw_kpi_cols: List of raw (unscaled) KPI column names
            position_name: Position name (e.g., 'Midfielder')

        Raises:
            ValueError: If cluster_label column is missing
        """
        if 'cluster_label' not in clustered_data.columns:
            raise ValueError("clustered_data must have 'cluster_label' column")

        self.clustered_data = clustered_data
        self.raw_kpi_cols = raw_kpi_cols
        self.position_name = position_name
        self.n_clusters = clustered_data['cluster_label'].nunique()

        # Results storage
        self.profiles = None
        self.f_statistics = {}
        self.z_scores = None
        self.cluster_names = {}

        logger.info(f"Initialized ClusterProfiler for {position_name}")
        logger.info(f"  Samples: {len(clustered_data)}")
        logger.info(f"  Clusters: {self.n_clusters}")
        logger.info(f"  KPIs: {len(raw_kpi_cols)}")

    def generate_profiles(self) -> pd.DataFrame:
        """
        Generate cluster profiles with descriptive statistics.

        Computes mean, median, std, min, max for each KPI per cluster.

        Returns:
            DataFrame with cluster profiles (mean values)
        """
        logger.info("\nGenerating cluster profiles...")

        profiles = []

        for cluster_id in range(self.n_clusters):
            cluster_data = self.clustered_data[
                self.clustered_data['cluster_label'] == cluster_id
            ]

            profile = {
                'cluster_id': cluster_id,
                'n_players': len(cluster_data)
            }

            # Calculate mean for each KPI
            for kpi in self.raw_kpi_cols:
                profile[f'{kpi}_mean'] = cluster_data[kpi].mean()
                profile[f'{kpi}_std'] = cluster_data[kpi].std()
                profile[f'{kpi}_median'] = cluster_data[kpi].median()

            profiles.append(profile)

        self.profiles = pd.DataFrame(profiles)

        logger.info(f"[OK] Profiles generated for {self.n_clusters} clusters")
        return self.profiles

    def calculate_f_statistics(self) -> Dict[str, float]:
        """
        Calculate F-statistics to identify discriminative KPIs.

        F-statistic measures how well a KPI separates clusters using ANOVA.
        Higher F-statistic = more discriminative KPI.

        Note: These are DESCRIPTIVE F-statistics, not inferential tests.
        We cannot use p-values after clustering (data snooping problem).

        Formula:
            F = (SSB / (k-1)) / (SSW / (n-k))
            where SSB = between-cluster sum of squares
                  SSW = within-cluster sum of squares
                  k = number of clusters
                  n = number of samples

        Returns:
            Dictionary mapping KPI names to F-statistics

        Reference:
            Fisher, R.A. (1925). Statistical Methods for Research Workers.
        """
        logger.info("\nCalculating F-statistics (discriminative power)...")

        f_stats = {}

        for kpi in self.raw_kpi_cols:
            # Group data by cluster
            groups = [
                self.clustered_data[
                    self.clustered_data['cluster_label'] == i
                ][kpi].values
                for i in range(self.n_clusters)
            ]

            # Calculate F-statistic using scipy
            f_stat, _ = f_oneway(*groups)
            f_stats[kpi] = f_stat

            logger.info(f"  {kpi}: F={f_stat:.2f}")

        # Sort by F-statistic
        self.f_statistics = dict(
            sorted(f_stats.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"\n[OK] Top 3 discriminative KPIs:")
        for i, (kpi, f_val) in enumerate(list(self.f_statistics.items())[:3], 1):
            logger.info(f"  {i}. {kpi} (F={f_val:.2f})")

        return self.f_statistics

    def calculate_z_scores(self) -> pd.DataFrame:
        """
        Calculate Z-scores for cluster profiles.

        Z-score shows how many standard deviations a cluster's mean is
        from the overall position mean.

        Formula:
            Z = (cluster_mean - position_mean) / position_std

        Interpretation:
            Z > 1: Cluster is strong in this KPI
            Z < -1: Cluster is weak in this KPI
            -1 < Z < 1: Cluster is average

        Returns:
            DataFrame with Z-scored cluster profiles
        """
        logger.info("\nCalculating Z-scores for cluster comparison...")

        z_scores = []

        for cluster_id in range(self.n_clusters):
            z_profile = {'cluster_id': cluster_id}

            for kpi in self.raw_kpi_cols:
                position_mean = self.clustered_data[kpi].mean()
                position_std = self.clustered_data[kpi].std()

                cluster_mean = self.profiles.loc[
                    self.profiles['cluster_id'] == cluster_id,
                    f'{kpi}_mean'
                ].values[0]

                # Calculate Z-score
                if position_std > 0:
                    z_score = (cluster_mean - position_mean) / position_std
                else:
                    z_score = 0.0

                z_profile[kpi] = z_score

            z_scores.append(z_profile)

        self.z_scores = pd.DataFrame(z_scores)

        logger.info("[OK] Z-scores calculated")
        return self.z_scores

    def get_cluster_characteristics(self, cluster_id: int, top_n: int = 3) -> Dict:
        """
        Identify defining characteristics of a cluster.

        Finds the top N KPIs that define this cluster based on:
        1. Absolute Z-score (distance from position mean)
        2. F-statistic (discriminative power)

        Args:
            cluster_id: Cluster to analyze
            top_n: Number of top KPIs to return

        Returns:
            Dictionary with top KPIs and their statistics
        """
        if self.z_scores is None:
            self.calculate_z_scores()

        cluster_z = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

        characteristics = []

        for kpi in self.raw_kpi_cols:
            z_score = cluster_z[kpi].values[0]
            f_stat = self.f_statistics.get(kpi, 0)
            mean_val = self.profiles.loc[
                self.profiles['cluster_id'] == cluster_id,
                f'{kpi}_mean'
            ].values[0]

            # Score = |Z-score| * sqrt(F-stat) to balance both factors
            importance_score = abs(z_score) * np.sqrt(f_stat)

            characteristics.append({
                'kpi': kpi,
                'z_score': z_score,
                'f_statistic': f_stat,
                'mean_value': mean_val,
                'importance_score': importance_score
            })

        # Sort by importance score
        characteristics = sorted(
            characteristics,
            key=lambda x: x['importance_score'],
            reverse=True
        )

        return characteristics[:top_n]

    def assign_tactical_names_midfielder(self) -> Dict[int, str]:
        """
        Assign tactical names to Midfielder clusters based on z-score profiles.

        Naming logic based on z-score patterns:
        - Balanced Midfielder: Average across all metrics (z-scores near 0)
        - High-Intensity Midfielder: Very high pressures, ball recoveries, tackles (z > 3.0)
        - Progressive Midfielder: Above average progressive passes and carries (z > 0.5)

        Returns:
            Dictionary mapping cluster_id to tactical name

        References:
            Sarmento, H. et al. (2019). Player typology in elite football.
            Bradley, P.S. et al. (2014). Yo-Yo test applications.
            D'Urso, P. et al. (2023). Fuzzy clustering of football players.
        """
        logger.info("\nAssigning tactical names (Midfielder)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            # Get cluster characteristics
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)

            # Get Z-scores for key KPIs
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Get key z-scores for decision logic
            z_pressures = z_row['pressures_per_90'].values[0] if 'pressures_per_90' in z_row.columns else 0
            z_prog_passes = z_row['progressive_passes_per_90'].values[0] if 'progressive_passes_per_90' in z_row.columns else 0
            z_prog_carries = z_row['progressive_carries_per_90'].values[0] if 'progressive_carries_per_90' in z_row.columns else 0
            z_recoveries = z_row['ball_recoveries_per_90'].values[0] if 'ball_recoveries_per_90' in z_row.columns else 0
            z_tackles = z_row['tackles_won_per_90'].values[0] if 'tackles_won_per_90' in z_row.columns else 0

            # Calculate intensity score (for detecting outliers with very high activity)
            intensity_score = z_pressures + z_recoveries + z_tackles

            # Calculate progression score
            progression_score = z_prog_passes + z_prog_carries

            # Decision tree for naming based on z-score profiles
            # High-Intensity: Very high z-scores (outliers with extreme activity levels)
            if z_pressures > 3.0 or intensity_score > 8.0:
                name = "High-Intensity Midfielder"
                justification = "Very high pressures, ball recoveries, and tackles (outlier profile)"
                literature = "Bradley et al. (2014)"

            # Progressive: Above average progression metrics
            elif z_prog_passes > 0.5 or progression_score > 0.8:
                name = "Progressive Midfielder"
                justification = "Above average progressive passes and carries"
                literature = "D'Urso et al. (2023)"

            # Balanced: Average or below average across metrics
            else:
                name = "Balanced Midfielder"
                justification = "Average across all metrics, balanced profile"
                literature = "Sarmento et al. (2019)"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {
                        'kpi': kpi['kpi'],
                        'z_score': round(kpi['z_score'], 2),
                        'mean': round(kpi['mean_value'], 2)
                    }
                    for kpi in top_kpis
                ],
                'intensity_score': round(intensity_score, 2),
                'progression_score': round(progression_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Literature: {literature}")
            logger.info(f"  Intensity score: {intensity_score:.2f}")
            logger.info(f"  Progression score: {progression_score:.2f}")
            logger.info(f"  Top 3 KPIs: {', '.join([k['kpi'] for k in top_kpis])}")

        self.cluster_names = names
        return names

    def assign_tactical_names_center_back(self) -> Dict[int, str]:
        """
        Assign tactical names to Center Back clusters.

        Uses domain knowledge and literature-based player typologies:
        - Ball-Playing Center Back: High progressive passes, pass completion
        - Aggressive Center Back: High pressures, blocks, aerial duels

        References:
            Dellal, A. et al. (2011). Physical and technical activity of soccer players.
            Hughes, M. et al. (2012). Moneyball and soccer.
        """
        logger.info("\nAssigning tactical names (Center Back)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Defensive/Aggressive KPIs
            z_pressures = z_row['pressures_per_90'].values[0] if 'pressures_per_90' in z_row.columns else 0
            z_blocks = z_row['blocks_per_90'].values[0] if 'blocks_per_90' in z_row.columns else 0
            z_aerial = z_row['aerial_duels_win_pct'].values[0] if 'aerial_duels_win_pct' in z_row.columns else 0
            z_interceptions = z_row['interceptions_per_90'].values[0] if 'interceptions_per_90' in z_row.columns else 0

            # Ball-playing KPIs
            z_prog_passes = z_row['progressive_passes_per_90'].values[0] if 'progressive_passes_per_90' in z_row.columns else 0
            z_pass_comp = z_row['pass_completion_pct'].values[0] if 'pass_completion_pct' in z_row.columns else 0

            # Calculate scores
            aggressive_score = z_pressures + z_blocks + z_aerial + z_interceptions
            ball_playing_score = z_prog_passes + z_pass_comp

            # Decision tree
            if aggressive_score > 1.5:
                name = "Aggressive Center Back"
                justification = "High pressures, blocks, and aerial dominance"
                literature = "Dellal et al. (2011)"
            elif ball_playing_score > 1.0:
                name = "Ball-Playing Center Back"
                justification = "High progressive passing and pass completion"
                literature = "Hughes et al. (2012)"
            elif aggressive_score > ball_playing_score:
                name = "Defensive Center Back"
                justification = "Focus on defensive actions"
                literature = "Dellal et al. (2011)"
            else:
                name = "Modern Center Back"
                justification = "Balanced defensive and distribution profile"
                literature = "General typology"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                    for kpi in top_kpis
                ],
                'aggressive_score': round(aggressive_score, 2),
                'ball_playing_score': round(ball_playing_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Aggressive score: {aggressive_score:.2f}, Ball-playing score: {ball_playing_score:.2f}")

        self.cluster_names = names
        return names

    def assign_tactical_names_full_back(self) -> Dict[int, str]:
        """
        Assign tactical names to Full Back clusters.

        Uses domain knowledge and literature-based player typologies:
        - Attacking Full Back: High progressive carries, xA, final third touches
        - Defensive Full Back: High tackles, interceptions, defensive duels

        References:
            Bush, M. et al. (2015). Evolution of match performance parameters.
            Bradley, P.S. et al. (2013). High-intensity running in FA Premier League.
        """
        logger.info("\nAssigning tactical names (Full Back)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Attacking KPIs
            z_prog_carries = z_row['progressive_carries_per_90'].values[0] if 'progressive_carries_per_90' in z_row.columns else 0
            z_xa = z_row['xa_per_90'].values[0] if 'xa_per_90' in z_row.columns else 0
            z_final_third = z_row['touches_final_third_per_90'].values[0] if 'touches_final_third_per_90' in z_row.columns else 0
            z_prog_passes = z_row['progressive_passes_per_90'].values[0] if 'progressive_passes_per_90' in z_row.columns else 0

            # Defensive KPIs
            z_tackles = z_row['tackles_interceptions_per_90'].values[0] if 'tackles_interceptions_per_90' in z_row.columns else 0
            z_def_duels = z_row['defensive_duels_win_pct'].values[0] if 'defensive_duels_win_pct' in z_row.columns else 0
            z_possession_won = z_row['possession_won_per_90'].values[0] if 'possession_won_per_90' in z_row.columns else 0

            # Calculate scores
            attacking_score = z_prog_carries + z_xa + z_final_third + z_prog_passes
            defensive_score = z_tackles + z_def_duels + z_possession_won

            # Decision tree
            if attacking_score > 1.5:
                name = "Attacking Full Back"
                justification = "High progressive carries, xA, and final third involvement"
                literature = "Bush et al. (2015)"
            elif defensive_score > 1.5:
                name = "Defensive Full Back"
                justification = "High tackles, interceptions, and defensive duels"
                literature = "Bradley et al. (2013)"
            elif attacking_score > defensive_score:
                name = "Overlapping Full Back"
                justification = "Moderate attacking contribution"
                literature = "Bush et al. (2015)"
            else:
                name = "Balanced Full Back"
                justification = "Balanced attacking and defensive profile"
                literature = "General typology"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                    for kpi in top_kpis
                ],
                'attacking_score': round(attacking_score, 2),
                'defensive_score': round(defensive_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Attacking score: {attacking_score:.2f}, Defensive score: {defensive_score:.2f}")

        self.cluster_names = names
        return names

    def assign_tactical_names_winger(self) -> Dict[int, str]:
        """
        Assign tactical names to Winger clusters.

        Uses domain knowledge and literature-based player typologies:
        - Creative Winger: High key passes, shot creating actions
        - Direct Winger: High successful dribbles, npxG+xA

        References:
            Di Salvo, V. et al. (2007). Performance characteristics of modern players.
            Lago-Penas, C. et al. (2010). Game-related statistics in elite soccer.
        """
        logger.info("\nAssigning tactical names (Winger)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Creative KPIs
            z_key_passes = z_row['key_passes_per_90'].values[0] if 'key_passes_per_90' in z_row.columns else 0
            z_sca = z_row['shot_creating_actions_per_90'].values[0] if 'shot_creating_actions_per_90' in z_row.columns else 0

            # Direct/Goal-threat KPIs
            z_dribbles = z_row['successful_dribbles_per_90'].values[0] if 'successful_dribbles_per_90' in z_row.columns else 0
            z_npxg_xa = z_row['npxg_plus_xa_per_90'].values[0] if 'npxg_plus_xa_per_90' in z_row.columns else 0
            z_touches_box = z_row['touches_penalty_area_per_90'].values[0] if 'touches_penalty_area_per_90' in z_row.columns else 0

            # Calculate scores
            creative_score = z_key_passes + z_sca
            direct_score = z_dribbles + z_npxg_xa + z_touches_box

            # Decision tree
            if direct_score > 2.0:
                name = "Direct Winger"
                justification = "High dribbles, goal threat (npxG+xA), and penalty area touches"
                literature = "Lago-Penas et al. (2010)"
            elif creative_score > 1.5:
                name = "Creative Winger"
                justification = "High key passes and shot creating actions"
                literature = "Di Salvo et al. (2007)"
            elif direct_score > creative_score:
                name = "Goal-Scoring Winger"
                justification = "Focus on goal involvement"
                literature = "Lago-Penas et al. (2010)"
            else:
                name = "Wide Playmaker"
                justification = "Balanced creative and direct profile"
                literature = "Di Salvo et al. (2007)"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                    for kpi in top_kpis
                ],
                'creative_score': round(creative_score, 2),
                'direct_score': round(direct_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Creative score: {creative_score:.2f}, Direct score: {direct_score:.2f}")

        self.cluster_names = names
        return names

    def assign_tactical_names_forward(self) -> Dict[int, str]:
        """
        Assign tactical names to Forward clusters.

        Uses domain knowledge and literature-based player typologies:
        - Target Man: High aerial duels, hold-up play
        - Mobile Striker: High dribbles, touches in penalty area

        References:
            Lago-Penas, C. et al. (2010). Game-related statistics in elite soccer.
            Castellano, J. et al. (2012). Match analysis and performance in soccer.
        """
        logger.info("\nAssigning tactical names (Forward)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Goal-scoring KPIs
            z_goals = z_row['non_penalty_goals_per_90'].values[0] if 'non_penalty_goals_per_90' in z_row.columns else 0
            z_npxg = z_row['npxg_per_90'].values[0] if 'npxg_per_90' in z_row.columns else 0
            z_shots_ot = z_row['shots_on_target_pct'].values[0] if 'shots_on_target_pct' in z_row.columns else 0
            z_conversion = z_row['conversion_rate'].values[0] if 'conversion_rate' in z_row.columns else 0

            # Mobile/Physical KPIs
            z_dribbles = z_row['successful_dribbles_per_90'].values[0] if 'successful_dribbles_per_90' in z_row.columns else 0
            z_touches_box = z_row['touches_penalty_area_per_90'].values[0] if 'touches_penalty_area_per_90' in z_row.columns else 0
            z_aerial = z_row['aerial_duels_win_pct'].values[0] if 'aerial_duels_win_pct' in z_row.columns else 0

            # Calculate scores
            clinical_score = z_goals + z_npxg + z_shots_ot + z_conversion
            mobile_score = z_dribbles + z_touches_box
            target_score = z_aerial

            # Decision tree
            if clinical_score > 2.0:
                name = "Clinical Striker"
                justification = "High goal scoring, npxG, and conversion rate"
                literature = "Castellano et al. (2012)"
            elif mobile_score > 1.5:
                name = "Mobile Striker"
                justification = "High dribbles and penalty area involvement"
                literature = "Lago-Penas et al. (2010)"
            elif target_score > 1.0:
                name = "Target Man"
                justification = "High aerial duels and hold-up play"
                literature = "Lago-Penas et al. (2010)"
            elif clinical_score > mobile_score:
                name = "Poacher"
                justification = "Focus on goal-scoring efficiency"
                literature = "Castellano et al. (2012)"
            else:
                name = "Complete Forward"
                justification = "Balanced goal-scoring and mobility profile"
                literature = "General typology"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                    for kpi in top_kpis
                ],
                'clinical_score': round(clinical_score, 2),
                'mobile_score': round(mobile_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Clinical score: {clinical_score:.2f}, Mobile score: {mobile_score:.2f}")

        self.cluster_names = names
        return names

    def assign_tactical_names_goalkeeper(self) -> Dict[int, str]:
        """
        Assign tactical names to Goalkeeper clusters.

        Uses domain knowledge and literature-based player typologies:
        - Ball-Playing Goalkeeper: High pass completion, progressive passes
        - Traditional Goalkeeper: Focus on shot-stopping

        References:
            Liu, H. et al. (2015). Goalkeeper performance analysis.
            West, J. (2018). Modern goalkeeper distribution patterns.
        """
        logger.info("\nAssigning tactical names (Goalkeeper)...")

        names = {}

        for cluster_id in range(self.n_clusters):
            top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
            z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]

            # Distribution KPIs
            z_pass_comp = z_row['gk_pass_completion'].values[0] if 'gk_pass_completion' in z_row.columns else 0
            z_prog_passes = z_row['progressive_passes_per_90'].values[0] if 'progressive_passes_per_90' in z_row.columns else 0

            # Shot-stopping/Sweeper KPIs
            z_xga = z_row['xga_per_90'].values[0] if 'xga_per_90' in z_row.columns else 0
            z_sweeper = z_row['sweeper_actions_per_90'].values[0] if 'sweeper_actions_per_90' in z_row.columns else 0
            z_cross = z_row['cross_claiming_rate'].values[0] if 'cross_claiming_rate' in z_row.columns else 0

            # Calculate scores
            distribution_score = z_pass_comp + z_prog_passes
            sweeper_score = z_sweeper + z_cross

            # Decision tree
            if distribution_score > 1.0:
                name = "Ball-Playing Goalkeeper"
                justification = "High pass completion and progressive distribution"
                literature = "West (2018)"
            elif sweeper_score > 1.0:
                name = "Sweeper Keeper"
                justification = "High sweeper actions and cross claiming"
                literature = "Liu et al. (2015)"
            elif z_xga > 1.0:
                name = "High-Workload Goalkeeper"
                justification = "Faces high expected goals against"
                literature = "Liu et al. (2015)"
            else:
                name = "Traditional Goalkeeper"
                justification = "Standard goalkeeper profile"
                literature = "General typology"

            names[cluster_id] = {
                'name': name,
                'justification': justification,
                'literature': literature,
                'top_3_kpis': [
                    {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                    for kpi in top_kpis
                ],
                'distribution_score': round(distribution_score, 2),
                'sweeper_score': round(sweeper_score, 2)
            }

            logger.info(f"\nCluster {cluster_id}: {name}")
            logger.info(f"  Justification: {justification}")
            logger.info(f"  Distribution score: {distribution_score:.2f}, Sweeper score: {sweeper_score:.2f}")

        self.cluster_names = names
        return names

    def assign_tactical_names(self) -> Dict[int, str]:
        """
        Assign tactical names based on position.

        Supports all 6 positions with position-specific naming logic.

        Returns:
            Dictionary mapping cluster_id to tactical name with metadata
        """
        if self.position_name == 'Midfielder':
            return self.assign_tactical_names_midfielder()
        elif self.position_name == 'Center Back':
            return self.assign_tactical_names_center_back()
        elif self.position_name == 'Full Back':
            return self.assign_tactical_names_full_back()
        elif self.position_name == 'Winger':
            return self.assign_tactical_names_winger()
        elif self.position_name == 'Forward':
            return self.assign_tactical_names_forward()
        elif self.position_name == 'Goalkeeper':
            return self.assign_tactical_names_goalkeeper()
        else:
            # Fallback for unknown positions
            logger.info(f"WARNING: Unknown position '{self.position_name}', using generic naming")
            names = {}
            for cluster_id in range(self.n_clusters):
                top_kpis = self.get_cluster_characteristics(cluster_id, top_n=3)
                z_row = self.z_scores[self.z_scores['cluster_id'] == cluster_id]
                avg_z = z_row[self.raw_kpi_cols].mean(axis=1).values[0] if len(z_row) > 0 else 0

                if avg_z > 0.5:
                    generic_name = f'{self.position_name} - Above Average'
                elif avg_z < -0.5:
                    generic_name = f'{self.position_name} - Below Average'
                else:
                    generic_name = f'{self.position_name} - Average'

                names[cluster_id] = {
                    'name': generic_name,
                    'justification': f'Mean Z-score: {avg_z:.2f}',
                    'literature': 'Auto-generated based on Z-scores',
                    'top_3_kpis': [
                        {'kpi': kpi['kpi'], 'z_score': round(kpi['z_score'], 2), 'mean': round(kpi['mean_value'], 2)}
                        for kpi in top_kpis
                    ],
                    'avg_z_score': round(avg_z, 2)
                }

            self.cluster_names = names
            return names

    def create_naming_justification_table(self) -> pd.DataFrame:
        """
        Create a table with cluster names and justifications.

        This table is thesis-ready and can be directly included in
        documentation or the final report.

        Returns:
            DataFrame with columns:
                - Cluster ID
                - Cluster Name
                - N Players
                - KPI Highlights
                - Tactical Interpretation
                - Supporting Literature
        """
        if not self.cluster_names:
            self.assign_tactical_names()

        table_data = []

        for cluster_id in range(self.n_clusters):
            cluster_info = self.cluster_names[cluster_id]
            n_players = self.profiles.loc[
                self.profiles['cluster_id'] == cluster_id,
                'n_players'
            ].values[0]

            # Format KPI highlights
            kpi_highlights = ', '.join([
                f"{kpi['kpi'].replace('_per_90', '').replace('_', ' ').title()}: {kpi['mean']}"
                for kpi in cluster_info['top_3_kpis'][:3]
            ])

            table_data.append({
                'Cluster ID': cluster_id,
                'Cluster Name': cluster_info['name'],
                'N Players': int(n_players),
                'KPI Highlights': kpi_highlights,
                'Tactical Interpretation': cluster_info['justification'],
                'Supporting Literature': cluster_info['literature']
            })

        table = pd.DataFrame(table_data)

        logger.info("\n" + "="*80)
        logger.info("NAMING JUSTIFICATION TABLE (THESIS-READY)")
        logger.info("="*80)
        logger.info(table.to_string(index=False))

        return table

    def save_profiles(self, output_dir: str, position_name_safe: str) -> Dict[str, str]:
        """
        Save all profiling results to files.

        Saves:
        1. Cluster profiles (CSV) - mean, std, median per cluster
        2. Z-scores (CSV) - for heatmap visualization
        3. F-statistics (JSON) - discriminative power of KPIs
        4. Tactical names (JSON) - names with justifications
        5. Naming justification table (CSV) - thesis-ready table

        Args:
            output_dir: Directory to save files
            position_name_safe: Safe filename version of position name

        Returns:
            Dictionary with paths to saved files
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        paths = {}

        # 1. Cluster profiles
        profiles_path = os.path.join(output_dir, f'{position_name_safe}_cluster_profiles.csv')
        self.profiles.to_csv(profiles_path, index=False, encoding='utf-8')
        paths['profiles'] = profiles_path

        # 2. Z-scores
        z_scores_path = os.path.join(output_dir, f'{position_name_safe}_z_scores.csv')
        self.z_scores.to_csv(z_scores_path, index=False, encoding='utf-8')
        paths['z_scores'] = z_scores_path

        # 3. F-statistics
        f_stats_path = os.path.join(output_dir, f'{position_name_safe}_f_statistics.json')
        with open(f_stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.f_statistics, f, indent=2, ensure_ascii=False)
        paths['f_statistics'] = f_stats_path

        # 4. Tactical names
        names_path = os.path.join(output_dir, f'{position_name_safe}_tactical_names.json')
        with open(names_path, 'w', encoding='utf-8') as f:
            json.dump(self.cluster_names, f, indent=2, ensure_ascii=False)
        paths['tactical_names'] = names_path

        # 5. Naming justification table
        table = self.create_naming_justification_table()
        table_path = os.path.join(output_dir, f'{position_name_safe}_naming_table.csv')
        table.to_csv(table_path, index=False, encoding='utf-8')
        paths['naming_table'] = table_path

        logger.info(f"\n[SUCCESS] Profiling results saved:")
        for key, path in paths.items():
            logger.info(f"  {key}: {path}")

        return paths


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("Cluster Profiling Module - Test")
    logger.info("="*60)

    np.random.seed(42)
    n_samples = 100

    # Generate synthetic clustered data
    synthetic_data = pd.DataFrame({
        'player_id': range(n_samples),
        'player_name': [f'Player_{i}' for i in range(n_samples)],
        'cluster_label': np.random.choice([0, 1, 2], size=n_samples),
        'kpi_1': np.random.randn(n_samples) * 10 + 50,
        'kpi_2': np.random.randn(n_samples) * 5 + 20,
        'kpi_3': np.random.randn(n_samples) * 15 + 100
    })

    raw_kpis = ['kpi_1', 'kpi_2', 'kpi_3']

    profiler = ClusterProfiler(synthetic_data, raw_kpis, 'Test Position')
    profiler.generate_profiles()
    profiler.calculate_f_statistics()
    profiler.calculate_z_scores()

    logger.info("\nTest completed successfully!")
