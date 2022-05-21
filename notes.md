## Things to investigate
* conv1d intialization as commented out in BENDER
* Reinsertion of LayerNorm in TransformerEncoderLayer (no tfixup)
* Projection head in convencoder
* Varying kernel filter sizes in convencoder
** Finding appropriate downsampling factor. Must avoid divergent loss and not simply memorize/overfit
* Initialization scheme (commmented out in BENDER)
* Use of x.clone() in official BENDER as workaround for multi-gpu
* L2 weight decay used in bendingcollegewav2vec
* SSL in BENDR does not sample negatives from masked states/features only, it samples any. In contrast to wav2vec 2 which samples only from previously masked inputs.

## TODO
- [] refactor masking in constrastive SSL forward function. Consider moving to data preprocessing and 
using huggingface numpy implementation or official wav2vec 2 implementation.
- []
