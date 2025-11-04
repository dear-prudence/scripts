from scripts.hestia import create_stream_snapshotFiles_lastgigyear

run = '09_18_lastgigyear'
halo = 'stream'
snaps = [108, 308]

for snap in range(snaps[1], snaps[0], -1):
    snap_ = '0' + str(snap) if snap < 100 else str(snap)

    output_path = '/z/rschisholm/storage/snapshots_stream/snapshot_' + snap_ + '.hdf5'
    create_stream_snapshotFiles_lastgigyear(snap, output_path)

    print('Completed snapshot ' + snap_ + '!')

