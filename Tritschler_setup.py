#Things specific to Tritschler et al. 2014
Ru           = 8314    #[J kmol-1 K-1]
Cp_air       = 0  #[J kg-1 K-1] 
Cv_air       = 0  #[J kg-1 K-1]
eps_on_k_air = 0  #[K]
sigma_air    = 0 

Cp_Heavy       = 0 #[J kg-1 K-1]
Cv_Heavy       = 0 #[J kg-1 K-1]
eps_on_k_Heavy = 0 #[K]
sigma_Heavy    = 0

M_N2           = 28.0140  #[kg kmol-1]
M_O2           = 31.9990  #[kg kmol-1]
M_SF6          = 146.0570 #[kg kmol-1]
M_Ac           = 58.0805  #[kg kmol-1]

gamma_N2       = 1.4 
gamma_O2       = 1.4
gamma_SF6      = 1.1
gamma_Ac       = 1.1

R_N2 = Ru/M_N2
R_O2 = Ru/M_O2
R_SF6 = Ru/M_SF6
R_Ac = Ru/M_Ac

Cp_N2       = gamma_N2/(gamma_N2-1.0)*R_N2  #[J kg-1 K-1] 
Cp_O2       = gamma_O2/(gamma_O2-1.0)*R_O2  #[J kg-1 K-1] 
Cp_SF6      = gamma_SF6/(gamma_SF6-1.0)*R_SF6  #[J kg-1 K-1] 
Cp_Ac       = gamma_Ac/(gamma_Ac-1.0)*R_Ac  #[J kg-1 K-1] 

Cv_N2       = Cp_N2/gamma_N2  #[J kg-1 K-1]
Cv_O2       = Cp_O2/gamma_O2  #[J kg-1 K-1]
Cv_SF6      = Cp_SF6/gamma_SF6  #[J kg-1 K-1]
Cv_Ac       = Cp_Ac/gamma_Ac  #[J kg-1 K-1]

M_Heavy = (0.8/M_SF6+0.2/M_Ac)**(-1)
M_Light = (0.767/M_N2+0.233/M_O2)**(-1)
