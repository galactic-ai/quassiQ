## About 
Find Changing-look AGNs using [SpenderQ](https://github.com/galactic-ai/SpenderQ/tree/main)

```
.
├── src
|   ├── pipeline <- pipeline to identify CLQ
|   ├── quality_cut <- filter out targets 
|   └── running_spenderq <- generate coadds and run SpenderQ
├── test
|   ├── overlap <- how many overlapped targets we have
|   ├── N_sigma <- check the distribution; pipeline
|   ├── latent <- produce plots for specific targets etc
|   ├── post_process <- study correlations to implement additional cuts
|   └── file <- inspect file
└── README.md
```
## Visual inspection

See the [visual inspection guide](VISUAL_INSPECTION.md) for starting the viewer on Lonestar6, joining as a reviewer, and resuming a session after a disconnect.


## Authors
Kanyuni Iemoto (Undergraduate student at UT Austin)

Prisha Desai (Undergraduate student at UT Austin)

ChangHoon Hahn (UT Austin)
