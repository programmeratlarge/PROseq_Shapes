#

# To do
# Add version number to each file (could be version directory instead) - Done (05/24/20)

# Version history:

# Label version 1 (LAB_V1) - original version:
# - Using directory depth[0]-window[100]-step[50]-andor[True]
# - 20Kb regions added to each end of annotation
# - Threshold of 1 TPM for filtering active annotations
# - Single label per window

# Label version 2 (LAB_V2) - revisions after defense:
# - Using directory depth[0]-window[100]-step[50]-andor[True]
# - 20Kb regions added to each end of annotation
# - Multiple labels per window
# - High confidence GENCODE annotations (only those that intersect w/ GRO-cap and poly-A sites)

# CNN version 1 - original version:
# - Window size: 8192 bins, bin size: 16bp, batch size: 128

# CNN version 2 - select center of window at random from within feature (instead of always midpoint)
# - Window size: 8192 bins, bin size: 16bp, batch size: 128

# CNN version 3 - 51K windows
# - Window size: 1024 bins, bin size: 50bp, batch size: 128

# CNN version 4 - 51K windows
# - Window size: 1024 bins, bin size: 50bp, batch size: 128
# - Reduce number of hidden layers to 5 (to compensate for smaller input size)

import os, subprocess, argparse, sys
# from keras.models import load_model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pyBigWig
import gc
from datetime import datetime
import set_up_globals

# Set up globals
labelVersion = set_up_globals.labelVersion
CNNVersion = set_up_globals.CNNVersion
data_folder = set_up_globals.data_folder
model_folder = set_up_globals.model_folder
gpuNumber = set_up_globals.gpuNumber

BATCH_SIZE = set_up_globals.BATCH_SIZE
WINDOW = set_up_globals.WINDOW
BINSIZE = set_up_globals.BINSIZE

os.environ["CUDA_VISIBLE_DEVICES"] = gpuNumber

# import pybedtools
# import time

# data_folder = '/local/workdir/prm88/a4_PROseq_shapes/data/'
##data_folder = '/local/workdir/prm88/a4_PROseq_shapes/jay-research/data'
##WINDOW = 200
##BINSIZE = 50
# WINDOW = 8192
# BINSIZE = 16

bws = {}
label_legend = {
    0: 'plus-neg',
    1: 'minus-neg',
    2: 'plus-genebody',
    3: 'minus-genebody',
    4: 'plus-aftergene',
    5: 'minus-aftergene',
    6: 'plus-genestart',
    7: 'minus-genestart',
    8: 'plus-geneend',
    9: 'minus-geneend',
    10: 'tss',
    11: 'plus-stable',
    12: 'minus-stable',
    13: 'plus-unstable',
    14: 'minus-unstable'
}

chrom_sizes = {"chr1": 249250621,
               "chr2": 243199373,
               "chr3": 198022430,
               "chr4": 191154276,
               "chr5": 180915260,
               "chr6": 171115067,
               "chr7": 159138663,
               "chr8": 146364022,
               "chr9": 141213431,
               "chr10": 135534747,
               "chr11": 135006516,
               "chr12": 133851895,
               "chr13": 115169878,
               "chr14": 107349540,
               "chr15": 102531392,
               "chr16": 90354753,
               "chr17": 81195210,
               "chr18": 78077248,
               "chr19": 59128983,
               "chr20": 63025520,
               "chr21": 48129895,
               "chr22": 51304566,
               "chrX": 155270560,
               "chrY": 59373566}

# Mouse amd horse
chrom_sizes_mm9 = {"chr1": 197195432,
                   "chr2": 181748087,
                   "chr3": 159599783,
                   "chr4": 155630120,
                   "chr5": 152537259,
                   "chr6": 149517037,
                   "chr7": 152524553,
                   "chr8": 131738871,
                   "chr9": 124076172,
                   "chr10": 129993255,
                   "chr11": 121843856,
                   "chr12": 121257530,
                   "chr13": 120284312,
                   "chr14": 125194864,
                   "chr15": 103494974,
                   "chr16": 98319150,
                   "chr17": 95272651,
                   "chr18": 90772031,
                   "chr19": 61342430,
                   "chrX": 166650296,
                   "chrY": 15902555}

chrom_sizes_equCab2 = {"chr1": 185838109,
                       "chr2": 120857687,
                       "chr3": 119479920,
                       "chr4": 108569075,
                       "chr5": 99680356,
                       "chr6": 84719076,
                       "chr7": 98542428,
                       "chr8": 94057673,
                       "chr9": 83561422,
                       "chr10": 83980604,
                       "chr11": 61308211,
                       "chr12": 33091231,
                       "chr13": 42578167,
                       "chr14": 93904894,
                       "chr15": 91571448,
                       "chr16": 87365405,
                       "chr17": 80757907,
                       "chr18": 82527541,
                       "chr19": 59975221,
                       "chr20": 64166202,
                       "chr21": 57723302,
                       "chr22": 49946797,
                       "chr23": 55726280,
                       "chr24": 46749900,
                       "chr25": 39536964,
                       "chr26": 41866177,
                       "chr27": 39960074,
                       "chr28": 46177339,
                       "chr29": 33672925,
                       "chr30": 30062385,
                       "chr31": 24984650,
                       "chrX": 124114077}


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iteration):
        # print '\r', self,
        print('\r', self, end='')
        # sys.stdout.write(str(self))
        sys.stdout.flush()
        self.update_iteration(iteration + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


def get_features(plus_bw, minus_bw, chrom, midpoint):
    global labelVersion
    global CNNVersion
    global data_folder
    global bws
    global label_legend
    global chrom_sizes
    global chrom_sizes_mm9
    global chrom_sizes_equCab2
    global BATCH_SIZE
    global WINDOW
    global BINSIZE

    # Start pulling features from the bigwigs
    halfwindow_binned = WINDOW // 2 * BINSIZE
    fullwindow_binned = WINDOW * BINSIZE

    #print('plus_bw.chroms:', plus_bw.chroms
    #print('chrom:', chrom)

    total_loci = plus_bw.chroms(chrom)
    end = min(total_loci, midpoint + halfwindow_binned)
    start = max(0, midpoint - halfwindow_binned)
    plus_arr = plus_bw.values(chrom, start, end, numpy=True)
    minus_arr = minus_bw.values(chrom, start, end, numpy=True)
    # Pad the features if necessary to have feature vectors of length fullwindow_binned
    if len(plus_arr) != fullwindow_binned and midpoint - halfwindow_binned < 0:
        plus_arr = np.pad(plus_arr, ((halfwindow_binned - midpoint), 0), 'constant')
        minus_arr = np.pad(minus_arr, ((halfwindow_binned - midpoint), 0), 'constant')
    if len(plus_arr) != fullwindow_binned and midpoint + halfwindow_binned > total_loci:
        plus_arr = np.pad(plus_arr, (0, midpoint + halfwindow_binned - total_loci), 'constant')
        minus_arr = np.pad(minus_arr, (0, midpoint + halfwindow_binned - total_loci), 'constant')
    # Stack the features so that the first row is the positive reads and second row is neg reads
    data = np.nan_to_num(np.vstack((plus_arr, np.abs(minus_arr))))
    # Scale the data to range between 0 and 1
    if np.max(data) > 0.:
        data = data / np.max(data)
    # Bin the features if necessary
    if BINSIZE:
        data = np.add.reduceat(data, np.arange(0, len(data[0]), BINSIZE), axis=1)

    return data


def write_preds(preds, locs, logfile):
    global labelVersion
    global CNNVersion
    global data_folder
    global bws
    global label_legend
    global chrom_sizes
    global chrom_sizes_mm9
    global chrom_sizes_equCab2
    global BATCH_SIZE
    global WINDOW
    global BINSIZE

    if len(preds[:, 0]) != len(locs):
        raise ValueError('Locs has different len from preds')
    chroms = [chrom for chrom, _ in locs]
    starts = [max(0, int(start) - 25) for _, start in locs]
    # Changed bws['tss'].chroms(chrom) to chrom_sizes(chrom) because of changes to pyBigWig (issue #97)
    ends = [min(chrom_sizes[chrom], int(start) + 25) for chrom, start in locs]
    logfile.write('\n'.join([chrom + ',' + str(start) for chrom, start in locs]) + '\n')
    for index, lbl in label_legend.items():
        bws[lbl].addEntries(chroms, starts, ends=ends, values=[float(i) for i in preds[:, index]])


def main(args):
    global labelVersion
    global CNNVersion
    global data_folder
    global model_folder
    global bws
    global label_legend
    global chrom_sizes
    global chrom_sizes_mm9
    global chrom_sizes_equCab2
    global BATCH_SIZE
    global WINDOW
    global BINSIZE

    chromo = args.chromo
    EPOCH_NUM = args.epoch
    cellType = args.celltype
    plusbwpath = args.plusbwpath
    minusbwpath = args.minusbwpath
    testMode = args.testmode

    # Set up chrom_sizes and informative bed path based on cell type
    informative_bed_path = data_folder + 'ref_files/bedbins/hg19_positions.50bp.sorted.' + chromo + '.bed'
    if cellType == 'MM9': 
        chrom_sizes = chrom_sizes_mm9
        informative_bed_path = data_folder + 'ref_files/bedbins/mm9_positions.50bp.sorted.' + chromo + '.bed'
    if cellType == 'equCab2': 
        chrom_sizes = chrom_sizes_equCab2
        informative_bed_path = data_folder + 'ref_files/bedbins/equCab2_positions.50bp.sorted.' + chromo + '.bed'

    output_folder = data_folder + 'bigwigs/' + labelVersion + '_' + CNNVersion + '/bigwigs_all_positions_50bp_' + cellType + '_' + chromo + '/'
    logfile = open(os.path.join(output_folder, 'progress-{}.log'.format(str(datetime.now()).split(' ')[0])), 'w+')

    LOAD_MODEL_FROM = data_folder + 'models/' + labelVersion + '_' + CNNVersion + '/' + model_folder + '/weights-{}.hdf5'.format(
        str(EPOCH_NUM).zfill(4))
    # //---LOAD_MODEL_FROM = '1-Dec-2018-multiclass-large-windows/weights-{}.hdf5'.format(str(EPOCH_NUM).zfill(4))

    print('\nProcessing for {}, using model file {}\n'.format(chromo, LOAD_MODEL_FROM))
    model = load_model(LOAD_MODEL_FROM)

    # print(model.summary())

    bed = pd.read_csv(filepath_or_buffer=informative_bed_path, sep='\t', header=None, names=['chrom', 'start', 'end'])

    with open(informative_bed_path) as f:
        numberOfRowsInFile = sum(1 for line in f)
    pbar = ProgressBar(numberOfRowsInFile)
    rowNumber = 0

    # if not os.path.isdir(output_folder):
    #    os.mkdir(output_folder)

    for v in label_legend.values():
        bws[v] = pyBigWig.open(os.path.join(output_folder, v + '.bw'), 'w')
        bws[v].addHeader(list(chrom_sizes.items()))

    # plus_bw = pyBigWig.open(data_folder + 'bigwigs/LAB_V1_GAN_V3/bigwigs_all_positions_50bp_PROseq_simulation_merged_chr7/plus-genebody_002.bw')
    # minus_bw = pyBigWig.open(data_folder + 'bigwigs/LAB_V1_GAN_V3/bigwigs_all_positions_50bp_PROseq_simulation_merged_chr7/minus-genebody_002.bw')

    # plus_bw = pyBigWig.open(data_folder + 'seq/simulated_PROseq/plus_genebody.bw')
    # minus_bw = pyBigWig.open(data_folder + 'seq/simulated_PROseq/minus_genebody.bw')

    # plus_bw = pyBigWig.open(data_folder + 'seq/ChROseq_merged/ChROseq_merged_0h_plus_normalized_NateWay.bw')
    # minus_bw = pyBigWig.open(data_folder + 'seq/ChROseq_merged/ChROseq_merged_0h_minus_normalized_NateWay.bw')

    # plus_bw = pyBigWig.open(data_folder + 'seq/G1/G1_plus.bw')
    # minus_bw = pyBigWig.open(data_folder + 'seq/G1/G1_minus.bw')

    plus_bw = pyBigWig.open(data_folder + plusbwpath)
    minus_bw = pyBigWig.open(data_folder + minusbwpath)

    features = []
    locs = []

    for i, row in bed.iterrows():
        # Show progress
        # if i % 10000 == 0: print('Row:', str(i))
        rowNumber += 1
        pbar.animate(rowNumber)

        chrom, binMidpoint, _ = row
        data = get_features(plus_bw, minus_bw, chrom, int(binMidpoint))
        features.append(data)
        locs.append((chrom, binMidpoint))
        if len(features) >= 1024:
            features = np.expand_dims(np.swapaxes(np.swapaxes(np.dstack(features), 0, 2), 1, 2), axis=3)
            preds = model.predict(features, batch_size=BATCH_SIZE)
            if testMode:
                print('Features:', features[0], '\nLength:', len(features))
                print('Locs:', locs[0:10], '\nLength:', len(locs))
                print('Preds:', preds[0:10], '\nLength:', len(preds))
            write_preds(preds, locs, logfile)
            features = []
            locs = []
            gc.collect()

    if len(features) > 0:
        features = np.expand_dims(np.swapaxes(np.swapaxes(np.dstack(features), 0, 2), 1, 2), axis=3)
        preds = model.predict(features, batch_size=BATCH_SIZE)
        write_preds(preds, locs, logfile)
        features = []
        locs = []
        gc.collect()

    for bw in bws.values():
        bw.close()

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chromo", help="Chromosome: chr1 - chr22, or chrX", default="chr21")
    parser.add_argument("-e", "--epoch", help="Epoch for the model you wish to use", type=int, default=650)
    parser.add_argument("-l", "--celltype", help="Cell type to use", default="K562")
    parser.add_argument("-p", "--plusbwpath", help="Path for plus bigwig file", default="seq/G1/G1_plus.bw")
    parser.add_argument("-m", "--minusbwpath", help="Path for minus bigwig file", default="seq/G1/G1_minus.bw")
    parser.add_argument("-t", "--testmode", help="Turn test mode on", action="store_true")
    args = parser.parse_args()

    main(args)
