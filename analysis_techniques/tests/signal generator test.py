# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp


# %%
# V1
def M(m1, m2):
    return np.power((m1*m2), (5/6))/np.power((m1+m2), (1/5))


def A(f, M_val, D):
    return np.sqrt(5*np.pi/24)*np.power((sp.G*M_val), 5/6)/(np.power(sp.c, 3/2)*D)*np.power(f, (-7/6))


def Psi(f, tc, pc, M_val):
    return 2*np.pi*f*tc-pc-(np.pi/4)+(3/128)*np.power((np.pi*M_val*f), (-5/3))


def h_tilde(f, m1, m2, D, tc, pc):
    M_val = M(m1, m2)
    return A(f, M_val, D)*np.exp(1j*Psi(f, tc, pc, M_val))


fs = np.linspace(10, 500, 1000)
hs = h_tilde(fs, 2e16*28.8, 18.1*2e16, 1e14, 2e-1, 3)

plt.plot(fs, np.real(hs))
plt.show()


# %%
# V2
def psi(f, tc, pc, M):
    return 2*np.pi*f*tc-pc-np.pi/4+(3/4)*np.power((8*np.pi*M*f), (-5/3))


def M(m1, m2):
    return np.power((m1*m2)/(m1+m2), (3/5))*np.power((m1+m2), (2/5))


def h(f, Q, D, m1, m2, tc, pc):
    M_val = M(m1, m2)
    return (Q/D)*np.power(M_val, (5/6))*np.power(f, (-7/6))*np.exp(1j*Psi(f, tc, pc, M_val))


fs = np.linspace(10, 500, 1000)
hf = h_tilde(fs, 2e30*29.1, 2e30*18, 1450*30e18, 20, 2)
ht = np.fft.ifft(hf)

# plt.plot(fs, np.real(hs))
plt.plot(ht)
plt.show()


# %%
# V3
Mpc = 3.08e+22
Msol = 2e+30
# Deff =


def Amp(chirpM):
    return (-1)*np.power((5/(24*np.pi)), (1/2))*(sp.G*Msol/(sp.c*sp.c*Mpc))*np.power((np.pi*sp.G*Msol/(sp.c*sp.c*sp.c)), (-1/6))*np.power((chirpM/Msol), (5/6))


def Wavefunc(f, TM, eta):
    v = np.power((sp.G*TM/(sp.c**3)*np.pi*f), (1/3))
    return (-1*np.pi/4)+((3/(128*eta)))*(np.power(v, -5)+(3715/756+55*eta/9)*np.power(v, -3)-16*np.pi*np.power(v, -2)+((15293365/508032)+(27145*eta/504)+(3085*eta*eta/72))*np.power(v, -1))


def fourier_template(f, m1, m2):
    TM = m1 + m2
    mu = m1*m2/TM
    eta = mu/TM
    chirpM = np.power(eta, (3/5))*TM
    return Mpc*Amp(chirpM)*np.power(f, (-7/6))*np.exp((-1)*1j*Wavefunc(f, TM, eta))


fs = np.linspace(10, 350, 1000)
h_f = fourier_template(fs, 40.1*Msol, 35*Msol)
h_t = np.fft.ifft(h_f)
plt.plot(h_t)


# %%
def phi(t, eta, M):
    return -1/eta * np.power((-t / 5 / M), 5/8)


def h_template(t, m1, m2, phi_0=0):
    TM = m1 + m2
    mu = m1*m2/TM
    eta = mu/TM
    chirpM = np.power(eta, (3/5))*TM

    const = - (sp.G * chirpM / np.power(sp.c, 2))
    variable = np.power((-t/(5 * sp.G * chirpM / np.power(sp.c, 3))), -1/4)
    cos_bit = np.cos(phi_0 + 2*phi(t, eta, TM))
    return const * variable * cos_bit

times = np.linspace(-1, 2, 1000)
s = h_template(times, 10*Msol, 4*Msol)
plt.plot(times, s)
plt.show()
# %%
