from scripts.local.archive.plots_old import create_image

base_path = '/Users/ursa/Desktop/mag_sims/output'

for i in range(62):
    input_path = base_path + '/snapshot_' + '{:03d}'.format(i) + '.hdf5'
    output_path = base_path + '_plots/snapshot_' + '{:03d}'.format(i) + '.png'
    create_image(filename=input_path, part_type='PartType2', bounds=[[-20, 20], [-20, 20]],
                 cmap='viridis', num_bins=400, export_path=output_path)
    print('Finished plotting snapshot ' + '{:03d}'.format(i) + '.')
