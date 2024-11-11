# Deformable Surface Reconstruction via Riemannian Metric Preservation

Official implementation of the paper.

**[Oriol Barbany](https://barbany.github.io/), [Adrià Colomé](https://www.iri.upc.edu/staff/acolome) and [Carme Torras](https://www.iri.upc.edu/people/torras/)**

**Institut de Robòtica i Informàtica Industrial (CSIC-UPC), Barcelona, Spain**

**Computer Vision and Image Understanding, 2024**

[`Publication`](https://www.sciencedirect.com/science/article/pii/S1077314224002364?via%3Dihub) | [`BibTeX`](#citation)  | [`arXiv`](https://arxiv.org/abs/2212.11596)

## Usage

### Installation

1. Clone this repository and navigate to the root directory:
```
git clone https://github.com/Barbany/riemannian_metric_preservation
cd riemannian_metric_preservation
```

2. Create a new conda environment, then install the package and its core dependencies:
```
conda create -n rmp python=3.10 -y
conda activate rmp
pip install -e .
```

3. Verify that CUDA is available in PyTorch by running the following in a Python shell:
```
# Run in Python shell
import torch
print(torch.cuda.is_available())  # Should return True
```
If CUDA is not available, consider re-installing PyTorch following the [official installation instructions](https://pytorch.org/get-started/locally/).

4. You can install optional dependencies such as those for development and for computing conformal
maps. The following command installs both of them, i.e., `dev` and `conformal`
optional dependencies:

```bash
pip install -e .[dev,conformal]
```

> [!NOTE]
> If you are using `zsh`, you may have to use quotes, e.g., `pip install -e ".[dev]"` to install the `dev` dependencies.

### Getting started

You can run the deformable reconstruction code as:
```
python -m rmp
```
followed by any argument that you want to change, e.g., regularization weights, dataset, etc.

### Data  

In this work, we use three datasets:
- [Deformable Surface Tracking (DeSurT)](https://ieeexplore.ieee.org/document/9010840)
- [Tracking Surface with Occlusion (TSO)](https://infoscience.epfl.ch/entities/publication/b2424835-ffb5-4306-9892-9ebaf34b61fa)
- [Texture-less Deformable Surfaces Dataset (TDS)](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/)

We put all the datasets into the same data structure:
```
PATH_TO_DATASET
├──model
│   ├──cam_extrinsic.tsv
│   ├──cam_intrinsic.tsv
│   └──mesh_vertices.tsv
└──ground_truth
    ├──1.png
    ├──2.png
   ... 
    └──NUM_IMAGES.png
```

To compute segmentation masks, you can run
`python rmp/data/compute_masks.py --data_root PATH_TO_DATASET`

To operate on cropped images, run the following on your dataset
`python rmp/data/compute_image_crops.py --data_root PATH_TO_DATASET`
The previous command saves cropped images and the associated camera parameters.
In case masks are available, cropped masks are also computed.

To estimate the depth using a monocular estimation you can run
`python rmp/data/compute_depth.py --data_root PATH_TO_DATASET`
which can then be used to compute point clouds with
`python rmp/data/compute_image_crops.py --data_root PATH_TO_DATASET`

For the parametrization, you can either use the values in the paper, which we obtain by associating
linearly spaced values in [0,1] maintaining the structure of the grid. Note that each dataset
(and even different sequences in a dataset) follow different conventions for the vertex order.
We make sure that the ordering is correct by checking that the vertex projections of the template
mesh have parameters associated that when used as Red and Green colors create a smooth gradient.
Alternatively, you can use the conformal map, which can be computed as
`python rmp/data/compute_conformal_maps.py --data_root PATH_TO_DATASET`

We also provide the code used to visualize the matches:
`python rmp/data/visualize_matches.py --data_root PATH_TO_DATASET --matches PATH_TO_MATCHES`


## License

The code in this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation

If you find this repository helpful, please consider citing our work:
```
@article{barbany24deformable,
  title = {Deformable surface reconstruction via Riemannian metric preservation},
  journal = {Computer Vision and Image Understanding},
  volume = {249},
  pages = {104155},
  year = {2024},
  issn = {1077-3142},
  doi = {https://doi.org/10.1016/j.cviu.2024.104155},
  url = {https://www.sciencedirect.com/science/article/pii/S1077314224002364},
  author = {Oriol Barbany and Adrià Colomé and Carme Torras},
}
```