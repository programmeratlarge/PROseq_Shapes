# PROseq_Shapes
Accurate de novo transcription unit annotation from run-on and sequencing data

Annotations of functional elements in Metazoan genomes are critical tools used to provide insight into the processes governing cell development, differentiation, and disease. Run-on and sequencing assays measure the production of nascent RNAs and can provide an effective data source for discovering functional elements. However, the accurate inference of functional elements from run-on and sequencing data remains an open problem because the signal is noisy and challenging to model. Here we investigated computational approaches that convert run-on and sequencing data into annotations representing transcription units, including genes and non-coding RNAs. We developed a convolutional neural network trained to identify different parts in the anatomy of a transcription unit, called CGAP, which we stitched together into transcript annotations using a hidden Markov model (HMM). Comparison against existing methods showed a small but significant performance improvement using our novel CGAP-HMM approach. We developed a voting system to ensemble the top three annotation strategies, resulting in large and significant improvements over the best performing method. Finally, we also report a conditional generative adversarial network (cGAN) as a generative approach to transcription unit annotation that showed promise for further development. Collectively our work provides tools for de novo transcription unit annotation from run-on and sequencing data that are accurate enough to be useful in many applications. 

## Citation

If you use this code or the resulting assemblies, please cite the following paper:

*Accurate de novo transcription unit annotation from run-on and sequencing data* <br />
Paul R. Munn, Jay Chia, Charles G. Danko <br />
Unpublished


## Prerequisites

* `Bash >= 4`
* `Python >= 3.5`
* Python modules: `tensorflow >= 2.0, scipy, numpy, matplotlib, getopt`
* `R >= 3.4.2`
* R libraries: `tunits, rqhmm`


## Installation

There is no need for installation of the code in this repository. However, the Tunits R library (written by Andre Martins: [https://github.com/andrelmartins](https://github.com/andrelmartins)) will need to be installed prior to running the HMM code. To install Tunits, follow these steps:
* There are several R packages that will need to be installed to get this working. The first is rqhmm:
* git clone [https://github.com/andrelmartins/QHMM.git](https://github.com/andrelmartins/QHMM.git)
* cd QHMM
* R CMD INSTALL rqhmm
* Then we need to set up the bigwig package, and T-units itself:
* git clone --recursive [https://github.com/andrelmartins/tunits.nhp](https://github.com/andrelmartins/tunits.nhp)
* cd bigWig; R CMD INSTALL bigWig; cd -
* make -C QHMM
* make -C tunits


## Data

The data required to run the programs below can be found in the data directory. This includes the following:
* A pre-trained model produced by the CNN 
* References file for hg19, mm9, and equCab2

Example PRO-seq bigwig files can be found in the GEO, under accession number [GSM1480327](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1480327)

## Program usage

### Suggested directory structure

Begin by creating a directory named "data" within the directory you have set up for the python and R scripts. Copy the "models" and "ref_files" directories (and all their sub-directories) into this data directory.

Next, create a "seq" directory under the data directory and copy the bigwig files produced by your PRO-seq assay into this.

Next, create a "bigwigs" directory under the data directory and within this create a "LAB_V2_CNN_V4" directory (this corresponds to the version numbers of the labels and CNN used to build the predictive model). Lastly, under this LAB_V2_CNN_V4 directory, create a directory for each of the chromosomes you will be making predictions for with the following names:

"bigwigs_all_positions_50bp_" + cell type + "_" + chromosome

For example, if you are making predictions for K562 cells, for chromosome 7 you would create a directory with the following name:

"bigwigs_all_positions_50bp_K562_chr7"

So your final directory structure should look like:

```
data
├── bigwigs
│   ├── LAB_V2_CNN_V4
│   │   ├── bigwigs_all_positions_50bp_K562_chr7
│   │   └── bigwigs_all_positions_50bp_K562_chr21
├── models
│   └── LAB_V2_CNN_V4
│   │   └── multiclass-50K-windows-random-center
├── ref_files
│   └── bedbins
└── seq
    └── G1
```

Once this structure is set up, the ```set_up_globals.py``` program needs to be edited to point to the data directory you have created. Specifically, line 31 of this code should be changed to:
```
data_folder = '<absolute path above data directory>/data/'
```

For example:
```
data_folder = 'C:/proseq_shapes/data/'
```

### Produce bigwig files of the predictions for each chromosome

```
write-bigwigs-all-positions-50bp.py
```

This code takes as input the chromosome you are making predictions for, the epoch number for the predictive model you are using, the cell type your PRO-seq assay was run on, and the pathways to the bigwigs files generated by your PRO-seq assay (for the plus and minus strands).

Program usage:

```
python write-bigwigs-all-positions-50bp.py \
-c <Chromosome you are making predictions for. Default: chr21> \
-e <Epoch for the model you are using. Default: 3610> \
-l <Cell type for your PRO-seq assay. Default: K562> \
-p <Path for plus bigwig file. Default=seq/G1/G1_plus.bw> \
-m <Path for minus bigwig file. Default=seq/G1/G1_minus.bw>
```

Example:

```
python write-bigwigs-all-positions-50bp.py \
-c chr7 -e 3610 -l K562 -p seq/G1/G1_plus.bw -m seq/G1/G1_minus.bw
```

This program writes the predictions as bigwig files into the appropriate directories under the data/bigwigs directory.

Multiple bigwig files are produced for each 'type' of region that the program is attempting to predict (for example, gene bodies, genes starts, gene ends, etc.), but the Tunits code as written will only use the gene body and gene start pedictions (plus-genebody.bw, minus-genebody.bw, plus-genestart.bw, and minus-genestart.bw). 

Prior to running the Tunits script the gene start files need to be converted to wig files. Additionally, we should filter out predictions of very small read counts (below 0.1 for our purposes) and then merge contiguous regions of read counts so that the Tunits script will run more efficiently.

### Produce thresholded / merged wig files:

Within each of the bigwig directories, run the following comand to convert the bigwig files to wig files:

```
bigWigToWig plus-genestart.bw plus-genestart.wig
bigWigToWig minus-genestart.bw minus-genestart.wig
```

Next, run the following awk script to apply the appropriate threshold:

```
cat plus-genestart.wig | awk 'BEGIN { OFS = "\t" } ; ($4 > 0.1) {print $0}' > plus-genestart-thresh01.wig
cat minus-genestart.wig | awk 'BEGIN { OFS = "\t" } ; ($4 > 0.1) {print $0}' > minus-genestart-thresh01.wig
```

Finally, use bedtools to merge contiguous regions within these thresholded files:

```
bedtools merge -i plus-genestart-thresh01.wig -c 4 -o max > plus-genestart-thresh01-merge.wig
bedtools merge -i minus-genestart-thresh01.wig -c 4 -o max > minus-genestart-thresh01-merge.wig
```

We are now ready to run Tunits' hidden Markov model to smooth the predictions and output the predicted gene bodies as bed files.

### Run Tunits

```
run.tunits.v2.R
```

This code takes as input the chromosome you are making predictions for, the cell type, and the path to your data directory.

Program usage:

```
Rscript --vanilla run.tunits.v2.R <chromosome> <cell type> <path to data directory>
```

Example:

```
Rscript --vanilla run.tunits.v2.R chr7 K562 C:/proseq_shapes/data/
```

Output files:

The program produces one bed file for each strand with the following names:

```
hmm3states1slot1covarContinuous.startpriors.LAB_V2_CNN_V4.gamma.<cell type>.<chromosome>.3000.preds.plus.bed and
hmm3states1slot1covarContinuous.startpriors.LAB_V2_CNN_V4.gamma.<cell type>.<chromosome>.3000.preds.minus.bed
```

So, for the example above we would get:

```
hmm3states1slot1covarContinuous.startpriors.LAB_V2_CNN_V4.gamma.K562.chr7.3000.preds.plus.bed and
hmm3states1slot1covarContinuous.startpriors.LAB_V2_CNN_V4.gamma.K562.chr7.3000.preds.minus.bed
```

These two files contain the predicted transcription unit annotations in .bed format for the plus and minus strands of the specified chromosome; i.e. Chromosome, Start postion, End position, Annotation identifier, Score, and Strand.

For example:
```
chr7	226250	227300	preds_plus_1	60	+
chr7	419950	420200	preds_plus_2	60	+
chr7	766300	779350	preds_plus_3	60	+
chr7	855350	951200	preds_plus_4	60	+
                    ⋮
```

## Building a consensus for CGAP-HMM, groHMM, and T-units:

In order to combine the strengths of our method and two existing, high performing, methods, we built a consensus annotation model from CGAP-HMM, groHMM ([Chae et al. 2015](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0656-3), [https://github.com/coregenomics/groHMM](https://github.com/coregenomics/groHMM)), and T-units ([Danko etal. 2018](https://www.nature.com/articles/s41559-017-0447-5), [https://github.com/andrelmartins](https://github.com/andrelmartins)). Errors made by each method do not appear to be correlated, so this combination should overcome their respective weaknesses. Specifically, we adopted the following approach:

* Remove small fragments (<101 bp) from each annotation dataset.
* Find the intersection of all annotations in each pair of datasets.
* Combine these intersections into a single dataset.
* Add back all transcription units from any method which does not intersect a transcription unit in the combined set by more than 10%.

Combining the three methods in this way overcomes the problem of annotations being merged and alleviates much of the problem with disassociation, while also achieving nearly the highest possible TUA score (0.9995 when four significant digits are considered).
