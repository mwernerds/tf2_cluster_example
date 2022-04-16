# OpenEducationalResource Wildfire Detection

Our wildfire detection example is an example of how to employ big data management strategies in a simple setup
using consumer hardware in a reliable and sensible way. The aim of this approach is to get a clear and rather
complete picture of a given problem and to analyze it in an efficient, distirbuted, but simple way.

This document is more about the tooling and setup of the processing system and not a guide on how to
create the highest numbers and scores for your models.

## Wildfire Detection - A Typology of models
Wilffire detection is a complex task and can be formulated as a data mining problem in at least three common
settings:

- Pixel-Wise: For each pixel, decide on its own whether it is a fire candidate TODO:ADD CITATIONS
- Pixel-Context-Wise: For each pixel and summary statistics of a fixed surrounding, e.g., a window of NxN pixels, decide if it is fire
- Patch-Wise: For a patch of NxN pixels, decide whether there is fire inside
- Semantic Segmenation: For an input image of mxn pixels, create an output image of m x n scores, where high scores
  indicate fire.


## Patch-Wise Fire Detection

We start with a dataset of Sentinel 2 images for which three binary masks have been defined from existing
geodata in a preprocessing step: one mask for cloud-covered pixels, one for nodata pixels, and one for fire
pixels. This document is not about how to come up with this type of data, we are going to publish some soon.

More concretely, the dataset is given in many files, each possibly having a different size and different projection
following the Sentinel 2 grid which foresees a different UTM projection for any of the TODO different tiles in which
the Level 2 data (that is preprocessed and georeferenced) is provided to users.

The three masks (fire, clouds, nodata) as a layer into these images and are, therefore, having the same projection.

Now, we create fire patches which contain (1) no clouds (2) no nodata pixels and (3) no pixels that have been used before.
In order to organize this efficiently, we create a new mask as eight-bit integer in which
- 0 means good data of class nofire
- 1 means good data of class fire
- 2 means used in a patch
- 3 means cloud
- 4 means nodata

Then, we localize all pixels of fire using something like `np.where (mask == 1)`, extract a patch from the mask and continue
to extract data patches for the fire class if the whole patch smaller or equal to 1. Afterwards, we mask the whole patch area with a value of
two.
> Note that it might be a good idea to mask a larger area to further reduce the risk of spatial autocorrelation between fire and nofire patches. 

We can easily loop through all candidate pixels as each patch extraction removes quite many of those. In addition, we randomize this list in order not to have a bias towards fire "appearing" on the right.

> Note that there is still a bias in this type of dataset that the center pixel belongs to the fire class. Maybe analyze what happens if you try to remove this structure from the dataset.

After extracting all fire patches, we want to extract nofire patches of the same amount. Therefore, we practically use the same approach, but now the list of locations to look at is taken to be random and we check that each patch is completely zero. This approach -- with a longer loop of trying -- brings us quite a few patches and due to the random sampling we don't induce much neighboring patch structures. However, depending on the amount of fire, this approach does not always generate the same amount of nofire pixels, a fact that we can take care of later as the order of magnitude is okay in our dataset.

With these preparations, we have two directories of PNG images, one for fires and one for nofire patches and we have introduced a few hardcoded biases and at least one hyperparameter to watch and vary later: the patch size.

As we want to be able to use [pretrained models](#@pretrained), we will build a handful of 3-channel composites out of the available spectral channels - first in the classical combinations used for visualization, later maybe with a [data-driven sampling method](#@cite:igarss).

For now, we do this by a two-step process: first scaling a part of the spectral information to the [0,1] range for each of the source image channels and then building composites according to literature.


