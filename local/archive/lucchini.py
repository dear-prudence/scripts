import numpy as np
import h5py


# This routine calculates temperatures following the procedure outlined in TNG documentation
def calc_T(u, e_abundance, x_h):
    gamma = 5 / 3  # needs from __future__ import division when run on a machine with Python 2.x
    k_b = 1.3807 * 1e-16
    m_p = 1.67262 * 1e-24
    unit_ratio = 1e10
    mu = 4 * m_p / (1 + 3 * x_h + 4 * x_h * e_abundance)
    print(mu)
    # internal energy per unit mass U is in (km/s)^2, and so needs the 10^6 to convert to (m/s)^2
    return (gamma - 1) * (u / k_b) * unit_ratio * mu


def gas_temp(U):
    xe = 1
    Xh = 0.76
    XHe = 1 - 0.76
    gamma = 5. / 3.
    m_p = 1.67262 * 1e-24
    k_b = 1.3807 * 1e-16

    # this statement is equivalent to other calculation for mu if Xhe = 1 - Xh, however since Scott's IC code did not
    # account for the electron abundance xe, need to set xe = 0 to account for that
    xe = 0
    mu = (1 + XHe / (1 - XHe)) / (1 + XHe / (4 * (1 - XHe)) + xe) * m_p
    # mu = 4./(1 + 3*Xh + 4*Xh*xe)*m_p
    print(mu)
    temp = (gamma - 1) * U * 1e10 / k_b * mu
    return temp


def lmc_temperatureProfile():
    part_type = 'PartType0'
    input_path = '/Users/ursa/smorgasbord/LMC24-80.hdf5'

    with h5py.File(input_path, 'r') as file:
        f = file[part_type]
        particles = {name: None for name in f.keys()}
        for key in f.keys():
            particles[key] = np.array(f[key])

    h = 0.677
    Z_solar = 0.0127
    metals_cutoff = 0.1
    X_H = 0.76

    # print('particles[\'Coordinates\'][0] = ' + str(particles['Coordinates'][0]))

    particles['Masses'] = particles['Masses'] * 1e10
    # print('particles[\'Masses\'][0] = ' + str(particles['Masses'][0]))

    distances = np.zeros(particles['Coordinates'].shape[0])
    for i in range(particles['Coordinates'].shape[0]):
        distances[i] = np.linalg.norm(particles['Coordinates'][i])
    particles['Distances'] = distances

    # particles['Temperature'] = calc_T(particles['InternalEnergy'], 1, X_H)
    particles['Temperature'] = gas_temp(particles['InternalEnergy'])

    total_HI_mass = np.sum(particles['Masses'] * X_H * particles['NeutralHydrogenAbundance'])
    total_HII_mass = np.sum(particles['Masses'] * X_H * (1 - particles['NeutralHydrogenAbundance']))
    avg_temp = np.average(particles['Temperature'], weights=particles['Masses'])
    average_fH0 = np.average(particles['NeutralHydrogenAbundance'], weights=particles['Masses'])

    radius_range = [0, round(1.2 * 150, 2)]
    log_temperature_range = [4.5, 6.5]
    bins = 500

    # radial binning is per 0.5 kpc, temperature binning is half number of radial bins
    hist, x_e, y_e = np.histogram2d(particles['Distances'], np.log10(particles['Temperature']),
                                    range=np.array([radius_range, log_temperature_range]),
                                    bins=[bins, int(bins / 2)],
                                    weights=particles['Masses'],
                                    density=True)

    print('hist.shape ' + str(hist.shape))
    column_averages = np.zeros(hist.shape[0])
    print('hist[0].shape ' + str(hist[0].shape))
    print('y_e[:-1] + abs(y_e[0] - y_e[1]) / 2 ' + str(np.array(y_e[:-1] + abs(y_e[0] - y_e[1]) / 2).shape))
    print('column_averages ' + str(column_averages.shape))
    for i in range(hist.shape[0]):
        try:
            column_averages[i] = np.average(y_e[:-1] + abs(y_e[0] - y_e[1]) / 2, weights=hist[i])
        except ZeroDivisionError:
            column_averages[i] = np.NaN

    print('--------------------------------------')
    print('M_HI, M_HII, T_avg, fH1_avg = ' + str(list([total_HI_mass, total_HII_mass, avg_temp, average_fH0])))
    print('--------------------------------------')

    return column_averages, x_e


def lucchini2020_orbits():
    a = np.array([[0.009795191451469343, 64.30868167202573],
                  [0.16295636687444348, 52.733118971061096],
                  [0.3410507569011576, 40.836012861736336],
                  [0.44078361531611754, 37.29903536977492],
                  [0.5298308103294747, 36.655948553054664],
                  [0.622439893143366, 39.54983922829582],
                  [0.8539626001780944, 51.76848874598071],
                  [1.0391807658058771, 63.02250803858521],
                  [1.3348174532502226, 78.77813504823152],
                  [1.5200356188780053, 86.4951768488746],
                  [1.7123775601068565, 91.96141479099678],
                  [1.9296527159394479, 96.14147909967846],
                  [2.079252003561888, 98.07073954983923],
                  [2.250222617987533, 98.71382636655949],
                  [2.421193232413179, 97.7491961414791],
                  [2.6028495102404277, 96.78456591639872],
                  [2.7666963490650045, 93.89067524115755],
                  [2.9091718610863757, 90.67524115755627],
                  [3.08726625111309, 84.2443729903537],
                  [3.2724844167408724, 76.84887459807074],
                  [3.429207479964381, 68.81028938906752],
                  [3.614425645592164, 57.556270096463024],
                  [3.756901157613535, 46.945337620578776],
                  [3.9100623330365094, 33.11897106109325],
                  [4.016918967052538, 25.723472668810288],
                  [4.081032947462155, 21.864951768488748],
                  [4.130899376669635, 21.221864951768488],
                  [4.2021371326803205, 21.864951768488748],
                  [4.248441674087266, 25.40192926045016],
                  [4.316117542297418, 28.617363344051448],
                  [4.398040961709706, 34.72668810289389],
                  [4.519145146927872, 41.157556270096464],
                  [4.67586821015138, 46.62379421221865],
                  [4.818343722172751, 49.19614147909968],
                  [4.971504897595725, 48.87459807073955],
                  [5.1246660730187, 45.337620578778136],
                  [5.252894033837934, 40.51446945337621],
                  [5.363312555654497, 34.08360128617363],
                  [5.44879786286732, 27.65273311897106],
                  [5.54853072128228, 19.935691318327976],
                  [5.641139804096171, 13.504823151125402],
                  [5.6981300089047195, 12.861736334405144]])
    return a
