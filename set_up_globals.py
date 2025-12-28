# 

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

# CNN version 5 - 51K windows
# - Add dialated convolutions

# Set up globals
labelVersion = 'LAB_V2'
CNNVersion = 'CNN_V5'
GANVersion = 'GAN_V1'
data_folder = '/local/workdir/prm88/a4_PROseq_shapes/data/'
#data_folder = '/local/workdir/prm88/a4_PROseq_shapes/jay-research/data'

model_folder = 'multiclass-50K-windows-random-center'
#model_folder = 'multiclass-large-windows-random-center'
#model_folder = 'multiclass-large-windows'

#WINDOW = 200
#BINSIZE = 50
BATCH_SIZE = 128
WINDOW = 1024
BINSIZE = 50

gpuNumber = "1"

print('Label version:', labelVersion)
print('CNN version:', CNNVersion)
print('GAN version:', GANVersion)
print('Data folder:', data_folder)
print('Model folder:', model_folder)

print('\nBATCH_SIZE:', BATCH_SIZE)
print('WINDOW:', WINDOW)
print('BINSIZE:', BINSIZE)

print('gpuNumber:', gpuNumber)
