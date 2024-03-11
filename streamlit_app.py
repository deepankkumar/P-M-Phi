# import libraries
import numpy as np
import streamlit as st

def interpolate_alpha(ec):
    e_cu = 0.002
    x = ec/e_cu
    x_range = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.00]
    y_range = [0, 0.290, 0.556, 0.762, 0.884, 0.912, 0.871, 0.801, 0.724]
    alpha = np.interp(x, x_range, y_range)
    return alpha

def interpolate_beta(ec):
    e_cu = 0.002
    x = ec/e_cu
    x_range = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.00]
    y_range = [0, 0.668, 0.677, 0.691, 0.728, 0.772, 0.827, 0.886, 0.943]
    beta = np.interp(x, x_range, y_range)
    return beta

steel = {'fy': 60, 'Es': 30000, 'eps_y': 0.002}
psteel = {'fy': 240, 'Es': 30000, 'Eps_y': 0.008}
concrete = {'fy': 5, 'Ec': 5000, 'eps_y': 0.002, 'eps_cu': 0.003, 'eps_ty': 0.0001, 'ft': 0.5}

# Create a sidebar for sliders
sidebar = st.sidebar
# Add a title to the sliders
sidebar.markdown("## Material and Beam Dimensions")

# Sliders in the sidebar
steel_fy = sidebar.slider("Steel fy (ksi)", min_value=40, max_value=80, value=60)
concrete_fy = sidebar.slider("Concrete fy (ksi)", min_value=3, max_value=10, value=5)
B = sidebar.slider("B (in)", min_value=10.0, max_value=20.0, value=14.0)
H = sidebar.slider("H (in)", min_value=15.0, max_value=30.0, value=24.0)
Cover = sidebar.slider("Cover (in)", min_value=1.0, max_value=5.0, value=3.0)

D1 = Cover
D = H - Cover 


# function to find fpso and fpso_u
def get_fpso(Ap, A_p, fpi):
    if Ap != 0 or A_p != 0:            
        P = Ap*fpi + A_p*fpi   
        M = Ap*fpi*(D-H/2) - A_p*fpi*(H/2-D1)
        Ag = B*H #+ As*30000/steel['fy']
        I = B*(H**3)/12 #+ As*30000/steel['fy']*(H/2)**2
        sigma_u = -(P/Ag) + (M/I)*(H/2)
        sigma_l = -(P/Ag) - (M/I)*(H/2)
        e_u = sigma_u/concrete['Ec']
        e_l = sigma_l/concrete['Ec']        
        e_pi = fpi/psteel['Es']
        e_ci = np.interp(H-D1, [0, H], [e_u, e_l])
        e_ci_u = np.interp(D1, [0, H], [e_u, e_l])
        e_pso = e_pi - e_ci
        e_pso_u = e_pi - e_ci_u
        f_pso = e_pso*psteel['Es']
        f_pso_u = e_pso_u*psteel['Es']
        c = 0
        phi = (e_u - e_l)/H
    else:
        P = 0
        M = 0
        c = 0
        phi = 0
    return f_pso, f_pso_u, e_pso, e_pso_u, P, M, c, phi

# Add title to the app
st.title("M-P-Phi created by Deepank")


P_list = []
M_list = []
Phi_list = []
C_list = []
alpha_1_list = []
beta_1_list = []
fp_list = []
f_p_list = []   
ecc_list = []

# CAse 1
# Ap = 4
# A_p = 4
# fpi = 0

# Add markdown to the slider and description of the slider
st.markdown("#### Top and Bottom Reinforcement Areas (in^2)")
# Add sliders to the main page



Ap = st.slider("Bottom Reinforcement (in^2)", min_value=1.0, max_value=10.0, value=4.0)
A_p = st.slider("Top Reinforcement (in^2)", min_value=1.0, max_value=10.0, value=4.0)
fpi = 0
# call get_fpso function
f_pso, f_pso_u, e_pso, e_pso_u, P, M, c, phi = get_fpso(Ap, A_p, fpi)

# append the values to the list
P_list.append(P)
M_list.append(M)
C_list.append(c)
fp_list.append(f_pso)
f_p_list.append(f_pso_u)
ecc_list.append(e_pso)
Phi_list.append(phi)

# Behaviour at cracking

def at_cracking(ecc,Ap,A_p,e_pso, e_pso_u,fpi):
    if fpi != 0:
        e_cp = ecc*(H-D)/H
        fp = psteel['Es']*(e_pso - e_cp)            
        sign = fp/abs(fp)

        if abs(fp) > psteel['fy']:
            fp = sign*psteel['fy']

        e_cp_ = ecc*(H-D1)/H
        fp_ = psteel['Es']*(e_pso_u - e_cp_)
        
        if abs(fp_) > psteel['fy']:
            fp_ = sign*psteel['fy']
                       
    if fpi == 0:
        e_cp = ecc*(H-D)/H
        fp = steel['Es']*(e_pso - e_cp)
        sign = fp/abs(fp)
        
        if abs(fp) > steel['fy']:
            fp = sign*steel['fy']

        e_cp_ = ecc*(H-D1)/H
        fp_ = steel['Es']*(e_pso_u - e_cp_)
        
        if abs(fp_) > steel['fy']:
            fp_ = sign*steel['fy']
   
    fp = -fp
    fp_ = -fp_
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)
    P = alpha*concrete['fy']*beta*B*H + Ap*fp + A_p*fp_
    M = A_p*fp_*(H/2-D1) - Ap*fp*(D-H/2) + alpha*concrete['fy']*beta*B*H*(H/2-beta*H/2)
    phi = ecc/H
    c = H
    return P, M, phi, c, fp, fp_

# Create a list of lists to store the values
table_data = []

# append ti table_data
table_data.append([phi*H, c, interpolate_alpha(phi*H), interpolate_beta(phi*H), f_pso, f_pso_u, phi*10**6, P, M])
# get the values of P, M, phi, c, fp, fp_ at cracking for ecc = 0.0005 to 0.003 with increment of 0.0005
ecc_arr = np.arange(0.0005, 0.0031, 0.00025)
for ecc in ecc_arr:
    P, M, phi, c, fp, fp_ = at_cracking(ecc,Ap,A_p,e_pso, e_pso_u,fpi)
    P_list.append(P)
    M_list.append(M)
    C_list.append(c)
    fp_list.append(fp)
    f_p_list.append(fp_)
    ecc_list.append(ecc)
    Phi_list.append(phi)
    # print ecc, c, alpha, beta, fp, fp_, phi*10^6, P, M in table format
    # Append the values to the table_data list
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)
    table_data.append([ecc, c, alpha, beta, fp, fp_, phi*10**6, P, M])
    
# Behaviour at yielding
def at_yielding(ecc,Ap,A_p,fpi,e_pso, e_pso_u):
    # get the value of alpha and beta
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)   
    
    # Define the variables
    if fpi != 0:
        c= D*ecc/(ecc + (psteel['Eps_y'] - e_pso))
    if fpi == 0:
        c= D*ecc/(ecc + (steel['eps_y'] - e_pso))
    if fpi != 0:
        f_p = psteel['Es']*(e_pso_u - ecc*(c-D1)/c)
        sign = f_p/abs(f_p)
        if abs(f_p) > psteel['fy']:
            f_p = sign*psteel['fy']
    if fpi == 0:
        f_p = steel['Es']*(e_pso_u - ecc*(c-D1)/c)
        sign = f_p/abs(f_p)
        if abs(f_p) > steel['fy']:
            f_p = sign*steel['fy']
                  
    f_p = -f_p
    fp = psteel['fy']
    if fpi == 0:
        fp = steel['fy']    
    P = alpha*concrete['fy']*beta*B*c - Ap*fp + A_p*f_p
    
    M_1 = alpha*concrete['fy']*beta*B*c*(H/2-beta*c/2) + A_p*f_p*(H/2-D1) + Ap*fp*(D-H/2)
    phi_1 = ecc/c
    C1 = c
    return M_1, phi_1, C1, fp, f_p, P

# Create a list of lists to store the values
table_data_1 = []

# get the values of P, M, phi, c, fp, fp_ at yielding for ecc = 0.0005 to 0.003 with increment of 0.0005
ecc_arr = np.arange(0.0005, 0.0031, 0.00025)
for ecc in ecc_arr:
    M_1, phi_1, C1, Fp, F_p, P_1 = at_yielding(ecc,Ap,A_p,fpi,e_pso, e_pso_u)
    M_list.append(M_1)
    Phi_list.append(phi_1)
    C_list.append(C1)
    P_list.append(P_1)
    fp_list.append(Fp)
    f_p_list.append(F_p)
    ecc_list.append(ecc)
    # print ecc, c, alpha, beta, fp, fp_, phi*10^6, P, M in table format
    # Append the values to the table_data list
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)
    table_data_1.append([ecc, C1, alpha, beta, Fp, F_p, phi_1*10**6, P_1, M_1])

 
# pure compression and pure tension
def pure_compression_tension(Ap,A_p,fpi,e_pso):
    if fpi != 0:
        Pc = -psteel['Es']*(e_pso - 0.002)*(Ap + A_p) + 0.85*concrete['fy']*(B*H - (Ap + A_p))
        Pt = -(Ap + A_p)*psteel['fy']
        M = 0
    if fpi == 0:
        Pc = steel['Es']*(0.002)*(Ap + A_p) + 0.85*concrete['fy']*(B*H - (Ap + A_p))
        Pt = -(Ap + A_p)*steel['fy']
        M = 0
    # make the other values NA
    c = np.nan
    phi = np.nan
    fp = np.nan
    f_p = np.nan
    return Pc, Pt, M, c, phi, fp, f_p 

# Behaviour at ultimate
def at_ultimate(c,Ap,A_p,fpi,e_pso, e_pso_u):
    # get the value of alpha and beta
    ecc= concrete['eps_cu']
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)   
    ecp_ = concrete['eps_cu']*(c-D1)/c
    ecp = concrete['eps_cu']*(D-c)/c
    ecc = concrete['eps_cu']
    if fpi != 0:
        f_p = psteel['Es']*(e_pso_u - ecc*(c-D1)/c)
        sign = f_p/abs(f_p)
        if abs(f_p) > psteel['fy']:
            f_p = sign*psteel['fy']
            
        fp = psteel['Es']*(e_pso + ecc*(D-c)/c)
        sign = fp/abs(fp)
        if abs(fp) > psteel['fy']:
            fp = sign*psteel['fy']
    if fpi == 0:
        f_p = steel['Es']*(e_pso_u - ecc*(c-D1)/c)
        sign = f_p/abs(f_p)
        if abs(f_p) > steel['fy']:
            f_p = sign*steel['fy']
            
        fp = steel['Es']*(e_pso + ecc*(D-c)/c)
        sign = fp/abs(fp)
        if abs(fp) > steel['fy']:
            fp = sign*steel['fy']
    f_p = -f_p
    P = alpha*concrete['fy']*beta*B*c - Ap*fp + A_p*f_p
    M = alpha*concrete['fy']*beta*B*c*(H/2-beta*c/2) + A_p*f_p*(H/2-D1) + Ap*fp*(D-H/2)
    phi = ecc/c
    return M, phi, c, fp, f_p, P

# Create a list of lists to store the values
table_data_2 = []

# call pure_compression_tension function
Pc, Pt, M, c, phi, fp, f_p = pure_compression_tension(Ap,A_p,fpi,e_pso)
M_list.append(M)
Phi_list.append(phi)
C_list.append(c)
P_list.append(Pt)
fp_list.append(fp)
f_p_list.append(f_p)
ecc_list.append(0)

table_data_2.append([np.nan, np.nan, np.nan, np.nan, fp, f_p, np.inf, Pt, M])

# get the values of P, M, phi, c, fp, fp_ at ultimate for c = 4 to 22 with increment of 2
c_arr = np.arange(2, 23, 2)

for c in c_arr:
    M_2, phi_2, C2, Fp, F_p, P_2 = at_ultimate(c,Ap,A_p,fpi,e_pso, e_pso_u)
    M_list.append(M_2)
    Phi_list.append(phi_2)
    C_list.append(C2)
    P_list.append(P_2)
    fp_list.append(Fp)
    f_p_list.append(F_p)
    ecc_list.append(ecc)
    # print ecc, c, alpha, beta, fp, fp_, phi*10^6, P, M in table format
    # Append the values to the table_data list
    alpha = interpolate_alpha(ecc)
    beta = interpolate_beta(ecc)
    table_data_2.append([ecc, C2, alpha, beta, Fp, F_p, phi_2*10**6, P_2, M_2])
    
Pc, Pt, M, c, phi, fp, f_p = pure_compression_tension(Ap,A_p,fpi,e_pso)
M_list.append(M)
Phi_list.append(phi)
C_list.append(c)
P_list.append(Pt)
fp_list.append(fp)
f_p_list.append(f_p)
ecc_list.append(0)

table_data_2.append([np.nan, np.nan, np.nan, np.nan, fp, f_p, 0, Pc, M])
# Case 1 is RC and Case 2 is PC
# get M, P and Phi for case 1 from the tables
M_RC_crack = [entry[-1] for entry in table_data]
P_RC_crack = [entry[-2] for entry in table_data]
Phi_RC_crack = [entry[-3] for entry in table_data]

# get M, P and Phi for case 1 from the tables for yielding
M_RC_yield = [entry[-1] for entry in table_data_1]
P_RC_yield = [entry[-2] for entry in table_data_1]
Phi_RC_yield = [entry[-3] for entry in table_data_1]


# GET M, P and Phi for case 1 from the tables for ultimate
M_RC_ultimate = [entry[-1] for entry in table_data_2]
P_RC_ultimate = [entry[-2] for entry in table_data_2]
Phi_RC_ultimate = [entry[-3] for entry in table_data_2]




import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot the P vs M for all the cases in the first subplot
axs[0].plot(M_RC_crack, P_RC_crack, label='RC, Cracking', color='r', linestyle='solid')
axs[0].plot(M_RC_yield, P_RC_yield, label='RC, Yielding', color='g', linestyle='solid')
axs[0].plot(M_RC_ultimate, P_RC_ultimate, label='RC, Ultimate', color='b', linestyle='solid')
axs[0].legend()
axs[0].set_xlabel('Moment (in-kips)')
axs[0].set_ylabel('Axial Force (kips)')
axs[0].set_title('P vs M')
axs[0].grid(axis='both', which='both', linestyle='--', linewidth=0.5)
axs[0].set_xticks(np.arange(0, 11000, 1000))
axs[0].set_yticks(np.arange(-500, 2500, 500))

# Plot the P vs Phi for all the cases in the second subplot
axs[1].plot(Phi_RC_crack, P_RC_crack, label='RC, Cracking', color='r', linestyle='solid')
axs[1].plot(Phi_RC_yield, P_RC_yield, label='RC, Yielding', color='g', linestyle='solid')
axs[1].plot(Phi_RC_ultimate, P_RC_ultimate, label='RC, Ultimate', color='b', linestyle='solid')
axs[1].legend()
axs[1].set_xlabel('Curvature (1/in) * 10^(-6)')
axs[1].set_ylabel('Axial Force (kips)')
axs[1].set_title('P vs Curvature')
axs[1].grid(axis='both', which='both', linestyle='--', linewidth=0.5)
axs[1].set_xticks(np.arange(0, 1600, 250))
axs[1].set_yticks(np.arange(-500, 2500, 500))

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Show the figure
plt.show()

# Streamlit display the figure
st.pyplot(fig)
