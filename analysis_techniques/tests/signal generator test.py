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
