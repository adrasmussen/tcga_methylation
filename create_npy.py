# /usr/bin/python

import pandas, tcga_tools, json
import numpy as np
import matplotlib.pyplot as plt

from progressbar import progressbar

# in mRNASeq and Meth450, the two cleaned and combined datasets are stored as UNION_chromosome.h5

# we import both, drop samples that do not appear in both, and then create either numpy array or csv as output


# the TGCA root folder
TCGA_root = tcga_tools.file_root

# chromosome must be a str since it includes X and Y
chromosome = tcga_tools.chromosome

# the name of the array information file
array_info_file = open(TCGA_root + 'output/ch_%s_npy_info.json' % chromosome, 'w')

# read in the two dataframes
print('Reading in meth450 and mRNA datasets: ', end='')
meth450_df = pandas.read_hdf(TCGA_root + 'Meth450/UNION_%s.h5' % chromosome)
mRNA_df = pandas.read_hdf(TCGA_root + 'mRNASeq/UNION_%s.h5' % chromosome)
print('Done')


# construct two sets from the rows
print('Finding intersection of samples: ', end='')
meth450_samples = set(map(tcga_tools.sample_parse, meth450_df.index.values))
mRNA_samples = set(map(tcga_tools.sample_parse, mRNA_df.index.values))

# find the intersection set
usable_samples = meth450_samples.intersection(mRNA_samples)
print('Done')

# init the lists for dropping last
wrong_meth450 = []
wrong_mRNA = []

# find the bad rows in both data frames
print('Removing bad samples: ', end='')
for i in meth450_df.index.values:
    if tcga_tools.sample_parse(i) not in usable_samples:
        wrong_meth450.append(i)

for j in mRNA_df.index.values:
    if tcga_tools.sample_parse(j) not in usable_samples:
        wrong_mRNA.append(j)

# drop the bad samples
meth450_df.drop(index=wrong_meth450, inplace=True)
mRNA_df.drop(index=wrong_mRNA, inplace=True)
print('Done')


# rename the rows
print('Sorting and renaming indices: ', end='')
meth450_df.rename(index=tcga_tools.sample_parse, inplace=True)
mRNA_df.rename(index=tcga_tools.sample_parse, inplace=True)

# sort the rows to ensure nothing got moved around
meth450_df.sort_index(inplace=True)
mRNA_df.sort_index(inplace=True)
print('Done')

# sanity checks
print('Checking sanity: ', end='')
if not all(list(meth450_df.index.values == mRNA_df.index.values)):
    raise ValueError
print('Done')

# adjust the outputs
mRNA_df = (mRNA_df - np.array(mRNA_df.min()).min())/np.array(mRNA_df.max()).max()
meth450_df = 1 - meth450_df


# save both to the top level for input into the neural net
print('Saving files: ', end='')
meth450_df.to_csv(TCGA_root + "output/meth450_%s_scrubbed.csv" % chromosome)
mRNA_df.to_csv(TCGA_root + "output/mRNA_%s_scrubbed.csv" % chromosome)

meth450_nd = meth450_df.to_numpy()
mRNA_nd = mRNA_df.to_numpy()

np.save(TCGA_root + 'output/meth450_%s' % chromosome, meth450_nd)
np.save(TCGA_root + 'output/mRNA_%s' % chromosome, mRNA_nd)

# the info dict
array_info = {}
array_info['samples'] = meth450_nd.shape[0]
array_info['inputs'] = meth450_nd.shape[1]
array_info['outputs'] = mRNA_nd.shape[1]

json.dump(array_info, array_info_file)

array_info_file.close()
print('Done')