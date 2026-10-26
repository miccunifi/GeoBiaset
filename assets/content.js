/* Edit this file to update the abstract, citation, and paper results.
   The abstract follows the supplied manuscript; displayed percentages use one decimal. */
window.GEOBIASET_CONTENT = {
  abstract: `World-level image geolocalization is an increasingly important task, yet the biases of current models remain poorly understood. Prior work has identified coarse dataset imbalances noting, for instance, that some cities are absent from training sets and that models tend to perform better in wealthier regions. However, no one has yet examined which visual cues drive these disparities. To do so, we introduce GeoBiaset, the first benchmark designed to measure how geolocalization models are influenced by the apparent race of a person inserted into the image foreground. The dataset consists of 9,541 images, pairing existing geotagged background images with counterfactual edits spanning seven races, holding the ground-truth coordinates fixed. Controlled masking experiments show that occlusion and background drift do not fully explain geolocalization degradation. Alongside the benchmark, we introduce the Person-Origin Perturbation Indicators (POPI), novel metrics that quantify model resilience to these demographic perturbations by measuring whether predictions are pulled toward the region associated with the inserted group. Our evaluation of state-of-the-art geolocalization and multimodal large language models reveals a 16.0–47.0% drop in continent-level performance when a subject from a demographically mismatched group is inserted. A landmark versus no-landmark analysis and gradient-weighted heatmaps show that models preferentially attend to geographic landmarks when present, but shift attention to facial features in their absence.`,
  citation: String.raw`@inproceedings{niccoli2026geobiaset,
  title  = {GeoBiaset: A Counterfactual Benchmark for
            Demographic Bias in World-Level Geolocalization},
  author = {Niccoli, Niccol{\`o} and Dani, Silvia and
            Mistretta, Marco and Bagdanov, Andrew D. and
            Bertini, Marco and Seidenari, Lorenzo},
  booktitle = {Advances in Neural Information Processing Systems},
  year   = {2026},
  note   = {Accepted at NeurIPS 2026, Track on Evaluations and Datasets},
  url    = {https://github.com/miccunifi/GeoBiaset}
}`,
  results: [
    { model: 'GeoCLIP', original: 81.9, edited: 55.3, attack: 29.3, help: 20.9, distraction: 30.7 },
    { model: 'Img2Loc', original: 70.3, edited: 37.5, attack: 34.5, help: 24.3, distraction: 36.2 },
    { model: 'G3', original: 81.6, edited: 47.8, attack: 28.1, help: 20.0, distraction: 29.4 },
    { model: 'DeepSeek-VL', original: 64.7, edited: 51.5, attack: 18.2, help: 19.3, distraction: 18.0 },
    { model: 'InternVL3.5', original: 58.0, edited: 43.2, attack: 24.1, help: 24.6, distraction: 24.1 },
    { model: 'Qwen3-VL', original: 79.3, edited: 65.3, attack: 17.8, help: 17.6, distraction: 17.8 },
    { model: 'GLM-4.1V', original: 82.8, edited: 69.2, attack: 16.2, help: 14.8, distraction: 16.4 }
  ]
};
