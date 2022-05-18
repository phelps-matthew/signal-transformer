## Things to investigate
* conv1d intialization as commented out in BENDER
* Reinsertion of LayerNorm in TransformerEncoderLayer (no tfixup)
* Projection head in convencoder
* Varying kernel filter sizes in convencoder
** Finding appropriate downsampling factor. Must avoid divergent loss and not simply memorize/overfit
* Initialization scheme (commmented out in BENDER)
* Use of x.clone() in official BENDER as workaround for multi-gpu
