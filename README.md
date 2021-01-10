# tcga_methylation
Deep learning setup for predicting cancer mRNA markers from chromosomal methylation data.


DATA CLEANING PIPELINE

The data inputs are CSV files downloaded and prepared by a third party, and thus the cleaning and combining steps are not generalizable.

The TCGA database is organized by cancer type first and then sample (from biopsy or similar).  Since we want to predict mRNA expression from methylation, it is necessary to ensure that we only include samples that have both measurements.

Based on genomic intuition, we expect that gene expression and methylation on each chromosome should be largely independent of other chromosomes, so the first step in the data cleaning process is to extract the specific chromosomes from the larger dataset.  This is controlled in the tgca_tools.py file.

The cleaner script is run first, which goes through each of the cancer CSV's and extracts the relevant columns from the CSV while dropping failed tests (which show up as NaNs).  This outputs a series of smaller .h5 tables, one for each cancer.

Then, the combiner script builds a single .h5 table for that chromosome.

Each of these scripts has to be run for both the mRNA and Meth450 data, producing a pair of cleaned and organized .h5 files.

Finally, there is a create_npy.py that finally creates a pair of sorted numpy arrays to facilitate data imports.  This script also checks that only the samples appearing in both mRNA and Meth450 datasets are included, and that they are sorted correctly in the numpy arrays.  It normalizes the output on the mRNA data (ensuring it is in [0,1]) and the Meth450 data (so that 0 is no methylation and 1 is max methylation).

There is an information dictionary saved in json format that keeps track of the array sizes.

The final output is a pair of numpy arrays (and scrubbed CSV's, for testing).


NEURAL NET ARCHITECHTURE

