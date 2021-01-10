# /usr/bin/python

import pandas, tcga_tools, warnings
import numpy as np
import matplotlib.pyplot as plt

from progressbar import progressbar

# silence the particular numpy warning in the checking logic
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'Mean of empty slice')

# Pre-cleaned data were stored in R dataframes (cancer.Rda), then decompressed and turned into .csv files
# saved in R via write.table(stuff, file="D:/TCGA/clean/Meth450/OV.csv", col.names=T, sep=",")

# this script reads in the csv files and removes columns that contain only NaN (does not find individual NaNs)
# then it selects a particular chromosome, transposes the data to get (samples, features), and saves it as an hdf5

# location of the files, allowed folders are Meth450, mRNASeq
TCGA_root = tcga_tools.file_root

# generated by R (oof)
TCGA_filetype = '.csv'

# allowed cancers are OV, ACC, etc
cancer_list = tcga_tools.cancer_list
#cancer_list = ['SKCM']

# chromosome must be a str since it includes X and Y
chromosome = tcga_tools.chromosome

# NaN cg sites
#
# format is cancer, site, site, ...
nan_cg_site_file = open(TCGA_root + 'Meth450/nan_cg_sites.csv', 'w')

# the union of sets of bad cg sites across all cancer types
nan_site_union = set()

### LOOP OVER CANCER LIST GOES HERE
for cancer in cancer_list:
    print('Reading in cancer: %s' % cancer)
    # create the dataframe
    D = pandas.read_csv(TCGA_root + 'Meth450/' + cancer + TCGA_filetype, sep=',')

    # select the desired chromosome
    D_chrom = D.loc[D['info.Chromosome'] == chromosome]

    # free the memory
    del D

    # drop unnecessary information
    # D_chrom.drop(columns=['info.Gene_Symbol', 'info.Chromosome', 'info.Genomic_Coordinate'], inplace=True)
    D_chrom.drop(columns=['info.Gene_Symbol', 'info.Chromosome', 'info.Genomic_Coordinate'], inplace=True)

    # Sort the dataframe on the genomic coordinate
    print('Sorting dataframe by CG site name: ', end='')
    D_chrom.sort_index(inplace=True)
    print("Done")


    # tranpose the matrix to get (samples, features)
    #
    # here, the N+1 sample is the genomic location
    D_chrom = D_chrom.transpose()

    # convert everything to float64
    #
    # anything it can't read gets turned into NaN, which will be removed later
    D_chrom = D_chrom.apply(lambda x: pandas.to_numeric(x, errors='coerce'))
    D_chrom = D_chrom.astype(np.float64)

    # init the nan set for this cancer
    nan_site = set()

    # new isnan checker
    print('Finding NaN sites:')

    # init an array for checking
    check_col = np.zeros(len(D_chrom.index))

    for cg in progressbar(D_chrom.columns):

        check_col = np.array(D_chrom[cg])
        nan_check = np.isnan(check_col)

        # first check if there are any NaNs:
        if nan_check.any():

            # if the whole column is NaN, add it to the list
            if nan_check.all():
                nan_site.add(cg)
                nan_site_union.add(cg)

            # otherwise, average the non-NaNs and set those
            else:
                col_avg = np.nanmean(check_col)
                nan_index = np.argwhere(np.isnan(check_col)).flatten()
                for i in nan_index:
                    D_chrom[cg][i] = col_avg

    # drop all of the entirely-NaN rows
    D_chrom.drop(columns=nan_site, inplace=True)

    # add the list of bad cg sites to the file
    nan_cg_site_file.write(cancer + ', ')
    for s in nan_site:
        nan_cg_site_file.write(s + ', ')

    nan_cg_site_file.write('\n')

    # error tolerant!
    nan_cg_site_file.flush()

    # print some useful information
    print('Cancer %s had %s NaN sites' % (cancer, len(nan_site)))
    print('Cancer %s had %s usable sites' % (cancer, D_chrom.shape[1]))

    # save the dataframe
    print('Saving %s chromosome %s dataframe: ' % (cancer, chromosome), end='')
    D_chrom.to_hdf(TCGA_root + 'Meth450/%s_%s.h5' % (cancer, chromosome), key='df')
    print('Done')

    print('\n')

# the last line of the bad site CSV is the union
nan_cg_site_file.write('UNION, ')
for u in nan_site_union:
    nan_cg_site_file.write(u + ', ')

nan_cg_site_file.write('\n')

nan_cg_site_file.close()

# plotting
#plt.plot(D_chrom.iloc[N, 0:100], D_chrom.iloc[0:N, 0:100].transpose())

print('Cleaning complete')