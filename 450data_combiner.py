# /usr/bin/python

import pandas, tcga_tools
import numpy as np
import matplotlib.pyplot as plt

from progressbar import progressbar


# Cleaned data files are stored individually as cancer_chromomsome.h5

# this script combines all of them in to a single UNION_chromosome.h5

# location of the files, allowed folders are Meth450, mRNASeq
# cleaned via 450data_cleaner.py => expected to be saved in HDF format
TCGA_root = tcga_tools.file_root

# allowed cancers are OV, ACC, etc
cancer_list = tcga_tools.cancer_list
#cancer_list = ['OV', 'UCS', 'CHOL']

# chromosome must be a str since it includes X and Y
chromosome = tcga_tools.chromosome

# init the dataframe
D = pandas.DataFrame()

for cancer in cancer_list:
    print('Reading in cancer: %s' % cancer)

    # read in the file for the given cancer type
    D_cancer = pandas.read_hdf(TCGA_root + 'Meth450/%s_%s.h5' % (cancer, chromosome))

    # concatenate each array to the end of D
    D = pandas.concat([D, D_cancer], axis=0) # CANNOT START WITH EMPTY FRAME

# save the file
print('Writing combined meth450 file: ', end='')
D.to_hdf(TCGA_root + 'Meth450/UNION_%s.h5' % chromosome, key='df')
print('Done')