# Waste Steel AI

**Industrial AI Prototype / Case Study**

Waste Steel AI is a Streamlit demonstration of an image-to-feature-to-matching workflow for a possible waste-steel inspection interface. It is useful for studying system boundaries, deterministic demos, and what evidence would be needed before an industrial claim could be made.

## What it is

- A runnable interface for uploading an image and visualizing a deterministic result
- A small, testable pipeline built with Python, Pillow, NumPy, scikit-learn, Matplotlib, and Streamlit
- An industrial AI practice project and case study

## What it is not

- It is **not** a waste-steel grading model trained on a labeled industrial dataset.
- It does **not** estimate real thickness, corrosion rate, or material purity from an image.
- Its relative matching score is a heuristic distance ratio, **not** a calibrated probability or confidence.
- It has not been validated for production, safety-critical, procurement, or quality-control decisions.

当前 Demo 不从真实工业标注数据中直接估计真实厚度、锈蚀率和纯度。界面中的这些数值是启发式代理值，只用于合成演示和可视化原型。

## Problem background

Image-assisted inspection could support recycling workflows, but a credible grading system would require representative data, independently verified labels, defined operating conditions, evaluation against suitable baselines, and field validation. This repository does not yet contain that evidence.

## Current prototype pipeline

1. Convert an uploaded image to grayscale.
2. Calculate mean brightness and pixel standard deviation.
3. Map those statistics to three bounded **proxy** values for demonstration.
4. Compare the proxy vector with three fixed demonstration centers using weighted Euclidean distance.
5. Display the nearest center, a heuristic relative matching score, and a synthetic PCA projection.

## Method

The distance is:

```text
sqrt(sum(weight_i * (feature_i - center_i)^2))
```

This is weighted Euclidean distance. It is not Mahalanobis distance because the prototype does not use an estimated inverse covariance matrix. The displayed PCA basis is fitted on fixed synthetic Gaussian samples and is used only for visualization.

## Limitations

- Brightness can change with lighting, camera exposure, shadows, and background.
- Pixel standard deviation is not a physical thickness measurement.
- The fixed centers and weights are demonstration assumptions, not learned or validated parameters.
- The three class labels are interface placeholders and do not establish compliance with an industrial standard.
- Locally saved review feedback is not used to retrain or update a model automatically.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Reproducibility and tests

Image-derived perturbation uses a SHA-256 content digest and a local `numpy.random.default_rng`; it does not depend on Python's process-randomized `hash()` or mutate NumPy's global random state. Synthetic PCA generation also uses a local fixed-seed generator.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests cover the weighted-distance formula, fixed-center classification, deterministic classification and feature extraction, and output ranges.

## Future validation

Before describing this as a grading model, future work would need to define a grading specification, collect governed and representative data, obtain reliable labels, separate training and evaluation data, report task-appropriate metrics and failure cases, and validate performance under real operating conditions.

## My role

I worked on requirement analysis, system design, AI-assisted implementation, documentation, demonstration materials, and review of the prototype's technical claims and limitations.

## License

No license has been selected for this repository. Source availability does not by itself grant reuse rights; a license decision is still required.
