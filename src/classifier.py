"""Nearest-center matching for the waste-steel demonstration."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class WasteSteelClassifier:
    """Match heuristic feature vectors to fixed demonstration centers."""

    def __init__(self) -> None:
        self.cluster_centers = {
            "I": np.array([8.3, 5.1, 0.92]),
            "II": np.array([4.2, 27.7, 0.74]),
            "III": np.array([2.2, 51.0, 0.48]),
        }
        self.weights = np.array([0.42, 0.35, 0.23])
        self.class_names = {
            "I": "I类（演示中心）",
            "II": "II类（演示中心）",
            "III": "III类（演示中心）",
        }
        self.feature_names = ["厚度代理值", "锈蚀代理值", "纯度代理值"]
        self.pca = self._initialize_pca()
        self.loadings = self.pca.components_

    @staticmethod
    def _initialize_pca() -> PCA:
        """Fit the fixed synthetic PCA projection used by the visualization."""

        rng = np.random.default_rng(42)
        sample_count = 400
        class_one = rng.multivariate_normal(
            [8.5, 10.0, 0.92],
            [[1.5, -0.5, 0.01], [-0.5, 5.0, -0.01], [0.01, -0.01, 0.001]],
            sample_count,
        )
        class_two = rng.multivariate_normal(
            [4.5, 30.0, 0.75],
            [[1.0, -0.2, 0.01], [-0.2, 10.0, -0.02], [0.01, -0.02, 0.005]],
            sample_count,
        )
        class_three = rng.multivariate_normal(
            [2.5, 55.0, 0.45],
            [[0.5, -0.1, 0.01], [-0.1, 15.0, -0.05], [0.01, -0.05, 0.01]],
            sample_count,
        )
        pca = PCA(n_components=2)
        pca.fit(np.vstack([class_one, class_two, class_three]))
        return pca

    def transform_to_pc(self, feature_vector: np.ndarray) -> np.ndarray:
        """Project one feature vector onto the synthetic PCA visualization."""

        vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        return self.pca.transform(vector)[0]

    def weighted_euclidean_distance(
        self, feature_vector: np.ndarray, center: np.ndarray
    ) -> float:
        """Calculate sqrt(sum(weight * difference^2))."""

        difference = np.asarray(feature_vector, dtype=np.float64) - center
        return float(np.sqrt(np.sum(self.weights * difference**2)))

    def calculate_distances(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Return weighted Euclidean distance to every fixed center."""

        return {
            label: self.weighted_euclidean_distance(feature_vector, center)
            for label, center in self.cluster_centers.items()
        }

    def classify(self, feature_vector: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Return nearest center, heuristic relative score, and PCA coordinates.

        The score is a distance ratio for this fixed set of centers. It is not a
        calibrated confidence or probability.
        """

        distances = self.calculate_distances(feature_vector)
        predicted_class = min(distances, key=distances.get)
        maximum_distance = max(distances.values())
        minimum_distance = distances[predicted_class]
        relative_score = (
            1.0 - (minimum_distance / maximum_distance)
            if maximum_distance > 0
            else 1.0
        )
        matching_score = round(relative_score * 100.0, 2)
        return predicted_class, matching_score, self.transform_to_pc(feature_vector)
