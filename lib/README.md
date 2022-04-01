`DerainedDataset` defines the training Dataset for synthetic rain images.
`DerainNet` defines the CNN structure for the derained task.
We first use a `guided filter` to generate the rain layer from the original rain image.
Then we use three Conv layers to learn the difference between the rain layer and the original image.
