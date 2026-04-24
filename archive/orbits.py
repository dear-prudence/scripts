import numpy as np


def besla2012():
    a = np.array([[0.018494860925613477, 59.70779861796644],
                  [0.14813309331629998, 48.80250856512397],
                  [0.21891876197665616, 43.349631264154226],
                  [0.28990767086696434, 39.259276464781365],
                  [0.4084838278845593, 34.19383311073688],
                  [0.46809128389756693, 33.80221822193832],
                  [0.6239184716334707, 38.46768480343766],
                  [0.8526508332849422, 51.88943731490621],
                  [1.165989199233494, 72.50984263399336],
                  [1.5623657162766391, 89.81801289123742],
                  [1.9337146507171479, 99.34126938040764],
                  [2.2923465536263867, 103.609546483944],
                  [2.5548458277684225, 103.4046803321526],
                  [2.8288136577434537, 100.08501248475699],
                  [3.054758724812728, 94.8207421171825],
                  [3.3516927007723134, 85.46611695023516],
                  [3.624121711863423, 71.83020730503455],
                  [3.883978863016085, 53.9125486324836],
                  [4.072440624818537, 37.36019975611171],
                  [4.237442657220836, 23.533824981127673],
                  [4.296382323906858, 18.665350444225083],
                  [4.343998606352709, 17.88490796121013],
                  [4.427936821322804, 20.6067011207247],
                  [4.608733523024215, 32.66778932698449],
                  [4.7413913245456145, 42.00569072643863],
                  [4.921316996690088, 48.227396782997495],
                  [5.029005284245979, 50.169676557691176],
                  [5.172173509087743, 49.969455896870095],
                  [5.267493176935138, 48.992509145810345],
                  [5.421955751698508, 44.509610359444856],
                  [5.528598803786076, 39.444631554497406],
                  [5.623250682306487, 33.990825155333596],
                  [5.72948725393415, 26.200801347192368],
                  [5.7762905754601945, 19.970268857789893],
                  [5.858951280413449, 14.127634864409742],
                  [5.906161082399397, 10.622147378201037],
                  [5.954009639393764, 11.39887346843966],
                  [6.050403577028047, 17.623831368677763],
                  [6.122757098890889, 22.68184193716971],
                  [6.230997038499507, 28.322397073340674],
                  [6.362580570234018, 30.458393821496998],
                  [6.529121421520238, 26.948260844317982],
                  [6.61195633238488, 22.27350328087799],
                  [6.682829104000931, 17.404564194878365],
                  [6.72992276871262, 13.120492422042828],
                  [6.753034086289996, 8.058765460774639],
                  [6.799866442134603, 2.0228790430288903],
                  [6.836188374658847, 5.525114685558327],
                  [6.89666686022879, 10.97288194646074]])
    return a


def package_data(sim_run, subject_halo, reference_halo):
    from archive.hestia import time_edges
    from archive.hestia import calc_distanceHalo

    snaps = [67, 127]
    time_e = time_edges(sim_run, snaps=np.arange(snaps[1], snaps[0], step=-1))
    distances = calc_distanceHalo(sim_run, snaps, subject_halo, reference_halo)

    np.savez('/z/rschisholm/storage/analytical_plots/orbits/' + sim_run + '_orbitalDistance_ '
             + (subject_halo[0:4] + subject_halo[5:] if len(subject_halo) != 3 else subject_halo) + '-'
             + (reference_halo[0:4] + reference_halo[5:] if len(reference_halo) != 3 else reference_halo) + '.npz',
             time_e=time_e, distances=distances, subject_halo=subject_halo, reference_halo=reference_halo)

    print('Done!')


def plotting(sim_run, subject_halo, reference_halo):
    import matplotlib.pyplot as plt

    input_path = ('/Users/dear-prudence/Desktop/smorgasbord/orbits/' + sim_run + '_orbitalDistance_'
                  + (subject_halo[0:4] + subject_halo[5:] if len(subject_halo) != 3 else subject_halo) + '-'
                  + (reference_halo[0:4] + reference_halo[5:] if len(reference_halo) != 3 else reference_halo)
                  + '.npz')
    output_path = ('/Users/dear-prudence/Desktop/smorgasbord/orbits/' + sim_run + '_orbitalDistance_'
                   + (subject_halo[0:4] + subject_halo[5:] if len(subject_halo) != 3 else subject_halo) + '-'
                   + (reference_halo[0:4] + reference_halo[5:] if len(reference_halo) != 3 else reference_halo)
                   + '.png')

    data = np.load(input_path)
    time = data['time_e'][:, 1]
    distances = data['distances']

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(time, distances, label='hestia')
    ax.plot(7 - besla2012()[:, 0], besla2012()[:, 1], label='besla+2021, model 2')
    ax.set_ylabel('Distance (kpc)')
    # ax.tick_params(axis='y')
    ax.set_xlabel(r'Lookback time $t$')
    plt.legend(loc='upper right')

    # Other formatting stuff
    # plt.xscale('log')
    plt.gca().invert_xaxis()
    x_lim = [7, 0]
    y_lim = [0, 200]
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    # plt.title('09_18 LMC mass accretion curves by particle type \n*stars and wind particles',
    # fontsize='small', loc='left')
    # -----------------------
    plt.savefig(output_path, dpi=240)
    plt.show()


# ------------------------------------
machine = 'dear-prudence'
# ------------------------------------
simulation_run = '09_18'
subject_halo_ = 'smc'
reference_halo_ = 'lmc'
# ------------------------------------

if machine == 'geras':
    package_data(simulation_run, subject_halo_, reference_halo_)

elif machine == 'dear-prudence':
    plotting(simulation_run, subject_halo_, reference_halo_)
