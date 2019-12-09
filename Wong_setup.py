#Things specific to Wong et al. 2019
#problem params (see table V in Wong and Lele 2019)
Ru = 8314 #[J kmol-1 K-1]
Cp_air       = 1040.50 #[J kg-1 K-1] 
Cv_air       = 743.697 #[J kg-1 K-1]
M_air        = 28.0135 #[kg kmol-1]
eps_on_k_air = 78.6 #[K]
sigma_air    = 3.711

Cp_Heavy       = 668.286 #[J kg-1 K-1]
Cv_Heavy       = 611.359 #[J kg-1 K-1]
M_Heavy        = 146.055 #[kg kmol-1]
eps_on_k_Heavy = 221 #[K]
sigma_Heavy    = 5.128

Ly = 0.025 #[m]
Lz = 0.025 #[m]

A = 1.06036 
B = -0.1561
C = 0.19300
D = -0.47635
E = 1.03587
F = -1.52996
G = 1.76474
H = -3.89411

lambda_3D = 1.02e-3 #[m]
eta_dot_3D_imp = 18.4 #[m s-1]

#calculate some constants
eps_ij = (eps_on_k_air*eps_on_k_Heavy)**0.5
sigma_ij = 0.5*(sigma_air+sigma_Heavy)
M_ij = 2.0/(1/M_air+1/M_Heavy)
tau_3D = lambda_3D/eta_dot_3D_imp
tau_c = tau_3D
