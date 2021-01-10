#!/usr/bin/python

# various tools for parsing or understanding the TCGA data


# the master list used for cleaning and repackaging data
#cancer_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

# SKCM appears to be missing ~400 nan sites and ~1500 usable sites on chromosome 12
cancer_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

# the chromosome we are currently interested in
# must be a string due to X and Y
chromosome = '12'

# root of the file system, with Meth450, mRNASeq, ch_gene_list, and output as subfolders
file_root = 'D:/TCGA/clean/'

def sample_parse(x):
    # takes in a TCGA barcode string and spits out a tuple of (patient, sample, portion)
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/

    # input should be TGCA string, but beta inputs handled differently
    if x.split('.')[0] == 'betas':
        patient, sample, portion = tuple([x.split('.')[i] for i in [3, 4, 5]])

    else:
        patient, sample, portion = tuple([x.split('.')[i] for i in [2, 3, 4]])

    # drop the vial and analyte information
    sample = sample[:-1]

    portion = portion[:-1]


    return (patient, sample, portion)