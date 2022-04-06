import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from h5py import File  # type: ignore

from .Variable import VariableError
from . import Units, RUnits

TWOPI = 2*np.pi
tfac = Units["Time"]


def _get_available_rads(sim, dr=50.):
    rmax = sim.get_coords(region='x', it=0, exclude_ghosts=3)[0]['x'].max()
    rads = []
    for rad in np.arange(1, rmax//dr + 1)*dr:
        try:
            if sim.get_data(f"psi4-r-l2-m2-r{rad:.2f}") is None:
                continue
        except VariableError:
            continue
        rads.append(rad)
    if len(rads) == 0:
        raise VariableError("no psi4 data in simulation {sim}")
    rad = rads[0]
    return np.array(rads)


def _get_data(sim, rad, ll, mm):
    dat = sim.get_data(f"psi4-r-l{ll}-m{mm}-r{rad:.2f}")
    ts = dat.times
    cc = dat.data
    icc = sim.get_data(f"psi4-i-l{ll}-m{mm}-r{rad:.2f}").data
    return ts, cc + 1j*icc


def _FFI(t, data, f0, order=2):

    n_points = len(t)
    delta_t = t[1] - t[0]

    ff = fftfreq(n_points, delta_t)

    ff[np.where(np.abs(ff) < f0)] = f0

    ct = fft(data)

    return ifft(ct/ff**order) * (-1j/TWOPI)**order


def get_f0(t0, tt, C22):

    if t0 < (tmin := tt.min()):
        raise ValueError(f't0 {t0} is smaller than minimum time of GW data: {tmin}')
    if t0 > (tmax := tt.max()):
        raise ValueError(f't0 {t0} is larger than maximum time of GW data: {tmax}')
    Phi = np.unwrap(np.angle(C22))
    dtPhi = np.diff(Phi)/np.diff(tt)
    dtPhit0 = interp1d((tt[:-1]+tt[1:])/2, dtPhi)(t0)

    return .5 * abs(dtPhit0)


def Planck_taper_window(tt, t0, width=200, end=True):

    t1 = t0 + width
    t2 = tt[-1] - width
    t3 = tt[-1]

    ww = np.zeros_like(tt)

    ww[(tt >= t1) & (tt <= t2)] = 1

    mask = (tt > t0) & (tt < t1)
    zz = (t1 - t0)/(tt[mask] - t0) + (t1 - t0)/(tt[mask] - t1)
    zz[zz > 100] = 100.
    ww[mask] = 1/(np.exp(zz) + 1)

    if end:
        mask = (tt > t2) & (tt < t3)
        zz = (t2 - t3)/(tt[mask] - t2) + (t2 - t3)/(tt[mask] - t3)
        zz[zz > 100] = 100.
        ww[mask] = 1/(np.exp(zz) + 1)
    else:
        ww[tt > t2] = 1

    return ww


def _isotropic2tortoise(rr, mm, aa=0):
    ar = rr*(1 + (mm + aa)/(2*rr))*(1 + (mm - aa)/(2*rr))
    return ar + 2*mm*np.log(ar/2/mm - 1)


def compute_PSD(sim, dist=100, post_merger=True, window_width=200, code_units=False, **kwargs):
    """
    Returns the PSD for the simulation sim.
    Arguments:
    - distance: of detector in Mpc
    - window_width: width of tapered part of the Planck window used to cut out the post merger frequencies
    - code_units: convert frequencies and PSD to Hz and Hz^-1
    - kwargs: keyword arguments for sim.get_data
    """

    hp = sim.get_data('h+', **kwargs)
    ts = hp.times
    hp = hp.data
    hx = sim.get_data('hx', **kwargs).data

    dist *= 3.0856776e+19*RUnits['Length']  # convert from Mpc

    data = (hp+1j*hx) / dist

    if post_merger:
        window = Planck_taper_window(ts, sim.t_merg-window_width, width=window_width, end=False)
        data = data*window
    # zero_mask = (window > 0.)
    # data = data[zero_mask]
    # ts = ts[zero_mask]

    tmp = data - np.polyval(np.polyfit(ts, data, 1), ts)

    PSD = np.abs(fft(tmp.real))**2
    PSD += np.abs(fft(tmp.imag))**2
    PSD = np.sqrt(0.5*PSD)

    frequencies = fftfreq(len(tmp), ts[1] - ts[0])
    frequencies = fftshift(frequencies)
    PSD = fftshift(PSD)

    # Kill the negative frequencies (they mirror the positive ones anyway)
    mask = frequencies >= 0
    frequencies = frequencies[mask]
    PSD = PSD[mask]

    if not code_units:
        frequencies /= tfac/1e3
        PSD *= tfac/1e3

    return frequencies, PSD


def _get_fit_func(rads, data, only_positive=False):
    def fitfunc(aas):
        if only_positive:
            aas[aas > 100] = 100
            tmp = 10**aas
        else:
            tmp = aas
        if len(tmp) == 1:
            return data[np.argmax(rads)] - tmp
        return data - sum(aa/rads**nn for nn, aa in enumerate(tmp))
    return fitfunc


def extract_strain(var, ll=2, mm=2, power=1, u_junk=200., n_points=3000):
    sim = var.sim

    if var.save:
        key = var.key
        for kk, item in dict(ll=ll, mm=mm, n_points=n_points, power=power, u_junk=u_junk).items():
            key += f":{kk}={item}"
        with File(sim.pp_hdf5_path, 'a') as hdf5:
            if key in hdf5:
                return hdf5[key].attrs['its'][:], hdf5[key].attrs['times'][:], hdf5[key][:], hdf5[key].attrs['restarts'][:]

    if sim.ADM_J is None or sim.ADM_M is None:
        raise ValueError("Could not determine ADM mass and/or ADM J")

    rads = _get_available_rads(sim)

    if power >= len(rads):
        raise ValueError(f"can't use power {power} with {len(rads)} radii")

    spin_parameter = sim.ADM_J/sim.ADM_M

    rad = max(rads)
    ts, cc22 = _get_data(sim, rad, 2, 2)
    trad = _isotropic2tortoise(rad, sim.ADM_M, spin_parameter)
    f0 = get_f0(u_junk+trad, ts, cc22)
    assert f0 > 0

    uu_loc = {}
    cc_loc = {}
    for rad in rads:
        t_junk = rad+u_junk

        ts, cc = _get_data(sim, rad, ll, mm)
        window = Planck_taper_window(ts, t_junk)
        cc *= window

        trad = _isotropic2tortoise(rad, sim.ADM_M, spin_parameter)
        uu_loc[rad] = ts - trad
        cc_loc[rad] = cc

    umin = uu_loc[rads.min()].min()
    umax = uu_loc[rads.max()].max()

    uu = np.linspace(max(0., umin), umax, n_points)

    AA = np.zeros((len(uu), len(rads)))
    Phi = np.zeros((len(uu), len(rads)))
    for ii, rad in enumerate(rads):

        AA_loc = np.abs(cc_loc[rad])
        AA[:, ii] = interp1d(uu_loc[rad], AA_loc)(uu)

        strt_ind = np.where(AA_loc > np.max(AA_loc)*.01)[0][0]
        phi = np.zeros_like(AA_loc)
        phi[strt_ind:] = np.unwrap(np.angle(cc_loc[rad][strt_ind:]))
        phi[:strt_ind] = phi[strt_ind]

        Phi[:, ii] = interp1d(uu_loc[rad], phi)(uu)

    rA_inf = np.zeros_like(uu)
    Phi_inf = np.zeros_like(uu)
    thresh = AA.max()*.01
    for ii, (aa, phi) in enumerate(zip(AA, Phi)):
        if all(aa < thresh):
            rA_inf[ii] = 0
            Phi_inf[ii] = 0
        else:
            fit = _get_fit_func(rads, rads*aa, only_positive=True)
            fit_pars, *_ = leastsq(fit, np.zeros(power+1)-1)
            rA_inf[ii] = 10**fit_pars[0]

            fit = _get_fit_func(rads, phi)
            fit_pars, *_ = leastsq(fit, np.zeros(power+1))
            Phi_inf[ii] = fit_pars[0]

    CC = rA_inf * np.exp(1j*Phi_inf)

    if mm > 0:
        F0 = f0*abs(mm)/2
    else:
        F0 = f0

    HH = _FFI(uu, CC, f0=F0, order=2)

    hp = HH.real
    hx = -HH.imag

    its = np.arange(len(uu)).astype(float)
    rsts = np.zeros(len(uu)).astype(float)
    if var.save:
        with File(sim.pp_hdf5_path, 'a') as hf:
            for pol, hh in zip("+ x".split(), (hp, hx)):
                name = f"h{pol}"
                for kk, item in dict(ll=ll, mm=mm, n_points=n_points, power=power, u_junk=u_junk).items():
                    name += f":{kk}={item}"
                if name in hf:
                    del hf[name]

                dset = hf.create_dataset(
                    name=name,
                    data=hh,
                    compression="gzip",
                    compression_opts=9,
                )
                dset.attrs['times'] = uu
                dset.attrs['its'] = its
                dset.attrs['restarts'] = rsts

    if 'h+' in var.key:
        return its, uu, hp, rsts
    return its, uu, hx, rsts
