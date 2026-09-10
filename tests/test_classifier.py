import numpy as np

from src.classifier import WasteSteelClassifier


def test_weighted_euclidean_distance_matches_formula():
    classifier = WasteSteelClassifier()
    vector = np.array([3.0, 4.0, 0.5])
    center = np.array([1.0, 1.0, 0.2])

    expected = np.sqrt(np.sum(classifier.weights * (vector - center) ** 2))

    assert classifier.weighted_euclidean_distance(vector, center) == expected


def test_each_fixed_center_matches_its_own_class():
    classifier = WasteSteelClassifier()

    for expected_class, center in classifier.cluster_centers.items():
        predicted_class, matching_score, _ = classifier.classify(center)
        assert predicted_class == expected_class
        assert matching_score == 100.0


def test_classification_is_deterministic():
    vector = np.array([4.0, 25.0, 0.7])

    first = WasteSteelClassifier().classify(vector)
    second = WasteSteelClassifier().classify(vector)

    assert first[0] == second[0]
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


def test_matching_score_is_bounded():
    classifier = WasteSteelClassifier()
    _, matching_score, _ = classifier.classify(np.array([100.0, 100.0, 1.0]))

    assert 0.0 <= matching_score <= 100.0
