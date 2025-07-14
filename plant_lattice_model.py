import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 12})

# DATA VALUES FOR (1) NO. INFECTED AND (2) PAIRS INFECTED
print("Loading data")
# Data taken from paper by Lanajeira et al.
data_t_og=[0,4,7,10,13,16,19,22,25,28,34,45,49,55]
data_t=[datat+30 for datat in data_t_og]
data_i=[0.0, 0.0, 0.015590200445434299, 0.053452115812917596, 0.0779510022271715, 0.2026726057906459, 0.24498886414253898, 0.2984409799554566, 0.38752783964365256, 0.48997772828507796, 0.6770601336302895, 0.8775055679287305, 0.955456570155902, 0.9755011135857461]
data_ii=[0.004166666666666667, 0.016666666666666666, 0.058333333333333334, 0.1125, 0.13333333333333333, 0.3, 0.3458333333333333, 0.4041666666666667, 0.49166666666666664, 0.5916666666666667, 0.7916666666666666, 0.925, 0.975, 0.9875]
data_cum=data_i

# CODE TO RUN THE LATTICE MODEL
REPS = int(input("How many replicates do you wish to run? (roughly 2000 runs per minute on a laptop)"))
THRESH=float(input("What threshold do you wish to use? (should be less than 0.5 - more replicates allow a lower threshold)"))

tot = 15 * 17  # (one row of 15 IP, then 16 rows of 15 S)
I0 = 1  # Starting no. of infecteds

# PAIR APPROXIMATION
def pairs(t, x):
    qis = x[3] / x[0]
    qips = x[7] / x[0]
    IP = 15 / tot
    I = 1 - x[0] - x[1] - IP
    sdot = -BETAS * x[0] * (L * qis + (1 - L) * I) - BETAP * x[0] * (
        LP * qips + (1 - LP) * IP
    )
    edot = (
        BETAS * x[0] * (L * qis + (1 - L) * I)
        + BETAP * x[0] * (LP * qips + (1 - LP) * IP)
        - GAMMA * x[1]
    )
    ssdot = (
        -2 * BETAS * (L * 3 / 4 * qis + (1 - L) * I) * x[2]
        - 2 * BETAP * (LP * 3 / 4 * qips + (1 - LP) * IP) * x[2]
    )
    sidot = (
        -BETAS * (L * (1 / 4 + 3 / 4 * qis) + (1 - L) * I) * x[3]
        - BETAP * (LP * (3 / 4 * qips) + (1 - LP) * IP) * x[3]
        + GAMMA * x[4]
    )
    sedot = (
        -BETAS * (L * 3 / 4 * qis + (1 - L) * I) * x[4]
        - BETAP * (LP * 3 / 4 * qips + (1 - LP) * IP) * x[4]
        - GAMMA * x[4]
        + BETAS * (L * 3 / 4 * qis + (1 - L) * I) * x[2]
        + BETAP * (LP * 3 / 4 * qips + (1 - LP) * IP) * x[2]
    )
    iidot = 2 * GAMMA * x[6]
    iedot = (
        -GAMMA * x[6]
        + BETAS * (L * (1 / 4 + 3 / 4 * qis) + (1 - L) * I) * x[3]
        + BETAP * (LP * (3 / 4 * qips) + (1 - LP) * IP) * x[7]
        + GAMMA * x[8]
    )
    sipdot = (
        -BETAS * (L * (3 / 4 * qis) + (1 - L) * I) * x[7]
        - BETAP * (LP * (1 / 4 + 3 / 4 * qips) + (1 - LP) * IP) * x[7]
    )
    eedot = -2 * GAMMA * x[8] + 2 * (
        BETAS * (L * 3 / 4 * qis + (1 - L) * I) * x[4]
        + BETAP * (LP * 3 / 4 * qips + (1 - LP) * IP) * x[4]
    )
    return [sdot, edot, ssdot, sidot, sedot, iidot, iedot, sipdot, eedot]

# To store all accepted parameter values
LstoreI = []
BstoreI = []
BPstoreI = []
GstoreI = []
DstoreI = []
objstoreI = []
objstoreII = []

# IClist: PS, PE, PSS, PSI, PSE, PII, PEI, PSIP
# Start with a row of IP along bottom row
IC = [
    1 - 15 / tot,
    0,
    1 - 0.75 * 15 / tot - 0.5 * 15 / tot,
    0,
    0,
    0,
    0,
    0.25 * 15 / tot,
    0,
]

for reps in range(REPS):
    if reps % 500 == 0:
        print(reps, 'simulations complete')
    L = np.random.rand()
    LP = L  
    DELTA = 30 * np.random.rand()
    rd_delt = round(DELTA, 1) # To easily do error calculations we round this to 0.1
    
    # Initially run for DELTA time units with no infections
    t0 = np.linspace(0, rd_delt - 0.1, int(rd_delt * 10))
    s0 = [240] * len(t0)
    e0 = [0] * len(t0)
    ii0 = [0] * len(t0)
    
    # Now run for rest of time for full model
    BETAP = 2 * np.random.rand()
    BETAS = 2 * np.random.rand()
    GS = 0.025 + 19.975 * np.random.rand()
    GAMMA = 1 / GS
    ts2 = np.linspace(rd_delt, 90, int((90 - rd_delt) * 10) + 1)
    xxpb = solve_ivp(pairs, [ts2[0], ts2[-1]], IC, t_eval=ts2, method="Radau")
    
    # Collect model predictions for P_I and P_{II}
    ts = np.concatenate((t0, ts2))
    xxp0 = np.concatenate((s0, xxpb.y[0]))
    xxp1 = np.concatenate((e0, xxpb.y[1]))
    xxp5 = np.concatenate((ii0, xxpb.y[5]))
    xxpi = 1 - xxp0 - xxp1 - 15 / tot
    xxpi = np.array([(xval * 255) / 240 for xval in xxpi])
    xxp5 = np.array([xval / (1 - 1.25 * 15 / 255) for xval in xxp5])
    
    # Calculate sum of squared differences between data and model
    objI = 0
    objII = 0
    for i in range(len(data_t)):
        if data_t[i] > (xxpb.t[-1]):
            Imodel = 1
            IImodel = 1
        else:
            Imodel = xxpi[data_t[i] * 10]
            IImodel = xxp5[data_t[i] * 10]
        objI += (Imodel - data_cum[i]) ** 2
        objII += (IImodel - data_ii[i]) ** 2
    # Store if below a maximal threshold
    if objI < 0.5 and objII < 0.5:
        objstoreI.append(objI)
        objstoreII.append(objII)
        LstoreI.append(L)
        BPstoreI.append(BETAP)
        BstoreI.append(BETAS)
        GstoreI.append(GAMMA)
        DstoreI.append(DELTA)

# NOW PLOT THE POSTERIORS FOR A GIVEN THRESHOLD

fig, ax = plt.subplots(1, 4,figsize=(10,3))

thresh=THRESH

print("Plotting posteriors")

choicelist=[]
for i in range(len(objstoreI)):
    if objstoreI[i]<thresh and objstoreII[i]<thresh:
        choicelist.append(i)
if len(choicelist)==0:
    print("No parameter sets accepted for chosen threshold. Suggest increasing no. of runs or threshold.")
Bs=np.array(BstoreI)
Bs=Bs[choicelist]
BPs=np.array(BPstoreI)
BPs=BPs[choicelist]
Ls=np.array(LstoreI)
Ls=Ls[choicelist]
Ds=np.array(DstoreI)
Ds=Ds[choicelist]
Gs=np.array(GstoreI)
Gs=1/Gs[choicelist]

# Histogram for L
ax[0].hist(Ls, range=(0,1),bins=20, label="$L$")
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Local ratio, $L$')

# Histograms for beta and beta_F
ax[1].hist(BPs, bins=20, label=r'$\beta_F$',alpha=0.5,color='y')
ax[1].hist(Bs, bins=20, label=r'$\beta$',alpha=0.6)
ax[1].legend(loc=1)
ax[1].set_xlabel(r'Transmission rates, $\beta,\beta_F$')

# Histogram for rho
ax[2].hist(Gs,range=(0,20),bins=20,label=r'$1/\rho$')
ax[2].set_xlabel(r'Latent period, $\rho$')

#Histogram for delta
ax[3].hist(Ds,range=(0,30),bins=20,label=r'$\delta$')
ax[3].set_xlabel('Delay, $\delta$')

ax[0].set_title('(a)')
ax[1].set_title('(b)')
ax[2].set_title('(c)')
ax[3].set_title('(d)')

plt.tight_layout()

# NOW RUN 100 DETERMINISTIC SIMULATIONS USING THESE POSTERIORS
print("Running 100 posterior deterministic checks")

fig, axs = plt.subplots(1, 2)

for i in range(100):
    choice=np.random.choice(choicelist)
    L=LstoreI[choice]
    LP=L
    DELTA=DstoreI[choice]
    rd_delt = round(DELTA, 1)

    t0 = np.linspace(0, rd_delt - 0.1, int(rd_delt * 10))
    s0 = [240] * len(t0)
    e0 = [0] * len(t0)
    ii0 = [0] * len(t0)
    ts2 = np.linspace(rd_delt, 90, int((90 - rd_delt) * 10) + 1)
    BETAP=BPstoreI[choice]
    BETAS=BstoreI[choice]
    GAMMA=GstoreI[choice]

    xxpb=solve_ivp(pairs,[ts2[0],ts2[-1]],IC,t_eval=ts2,method='LSODA')
    ts=np.concatenate((t0,ts2))
    xxp0=np.concatenate((s0,xxpb.y[0]))
    xxp1=np.concatenate((e0,xxpb.y[1]))
    xxp5=np.concatenate((ii0,xxpb.y[5]))
    xxpi=1-xxp0-xxp1-15/tot
    xxpi=np.array([(xval*255)/240 for xval in xxpi])
    xxp5=np.array([xval/(1-1.25*15/255) for xval in xxp5])
    # Occasionally solver does not run fully - catch these cases
    if len(xxpi)==len(ts):
        axs[0].plot(ts,xxpi,c='r',alpha=0.05)
        axs[1].plot(ts,xxp5,c='r',alpha=0.05)

axs[0].set_xlabel('Months')
axs[0].set_ylabel('Proportion Infected')
axs[0].set_ylim(0,1)
axs[0].set_xlim(0,90)
axs[0].scatter(data_t,data_cum)
axs[0].set_title('(a)')
axs[1].set_xlabel('Months')
axs[1].set_ylabel('Pairs Infected')
axs[1].set_ylim(0,1)
axs[1].set_xlim(0,90)
axs[1].scatter(data_t,data_ii)
axs[1].set_title('(b)')
plt.tight_layout()

# NOW RUN 100 STOCHASTIC SIMULAITIONS USING THESE POSTERIORS

def run_stochastic(L, BETA, GAMMA, DELTA,BETAF):
    tot=17*15
    I0=1 
    # Function to count noumbers in each compartment
    def count_type(sirtype):
        return len(np.where(grid==sirtype)[0])

    def local_inf(site_x,site_y):
        localinfs=0

        
        if site_x>0:
            if grid[site_x-1,site_y]==2:
                localinfs=localinfs+1
        if site_x<16:
            if grid[site_x+1,site_y]==2:
                localinfs=localinfs+1
        if site_y>0:
            if grid[site_x,site_y-1]==2:
                localinfs=localinfs+1
        if site_y<14:
            if grid[site_x,site_y+1]==2:
                localinfs=localinfs+1
        return localinfs
    
    def local_f(site_y):
        localfs=0

        if grid[0,site_y]==3:
                localfs=localfs+1
        return localfs

    # Function to check the current scale
    def findscale():
        S=count_type(0)
        I=count_type(2)
        E=count_type(1)
        localinf=0
        localf=0
        if S>0:
            gridchoice=np.where(grid==0)
            for i in range(len(gridchoice[0])):
                localinf+=local_inf(gridchoice[0][i],gridchoice[1][i])
                if gridchoice[0][i]==1:
                    localf+=local_f(gridchoice[1][i])
            QIS=localinf/S*tot/4
            QFS=localf/S*tot/4               
        else:
            QIS=0
            QFS=0
      #Set relative parameter values
        scale=GAMMA*E+BETA*S*(L*QIS+(1-L)*I)+BETAF*S*(LP*QFS+(1-LP)*15)
        return scale
    
    init_inf=I0
    grid=np.zeros((17,15))
    grid[0,:]=3
    tsteps=[0]
    susceptibles=[240]
    infecteds=[0]
    exposeds=[0]
    current_t=DELTA
    rows=np.zeros(16)
    r4=np.zeros(16)
    r11=np.zeros(16)
    rowfind=np.zeros(16)
    r4find=np.zeros(16)
    r11find=np.zeros(16)
    counters=np.zeros(3)
    LP=L

  # Main run
    while current_t<120:
        
        
        # Find tau-leap
        scale=findscale()
        dt = -np.log(np.random.rand()) / scale
        
        flagged=0   # Used to break out of 2nd loop

        #Find event
        if np.random.rand()<GAMMA*exposeds[-1]/scale: #Event is E -> I
            gridchoice=np.where(grid==1)
            fr=np.random.randint(0,len(gridchoice[0]))
            grid[gridchoice[0][fr],gridchoice[1][fr]]=2
        else: #Event is S -> E
            # Transmission is global due to either infected or founder
            rand2=np.random.rand()
            EVENT1=susceptibles[-1]*((1-L)*BETA*infecteds[-1]+(1-LP)*BETAF*15)
            gridchoice=np.where(grid==0)
            localinf=0
            for i in range(len(gridchoice[0])):
                localinf+=local_inf(gridchoice[0][i],gridchoice[1][i])
            QIS=localinf/susceptibles[-1]*tot/4
            EVENT2=BETA*L*susceptibles[-1]*QIS
            
            if rand2<EVENT1/(scale-GAMMA*exposeds[-1]): 
                fr=np.random.randint(0,len(gridchoice[0]))
                grid[gridchoice[0][fr],gridchoice[1][fr]]=1
                counters[0]+=1
            # Transmission is local
            # And due to an infected
            # Need to find local densities
            
            elif rand2<(EVENT1+EVENT2)/(scale-GAMMA*exposeds[-1]): 
                findx=[i for i in range(len(gridchoice[0]))]
                np.random.shuffle(findx)
                for f in findx:
                    if local_inf(gridchoice[0][f],gridchoice[1][f])>0:
                        grid[gridchoice[0][f],gridchoice[1][f]]=1
                        counters[1]+=1
                        break
            # or due to a founder
            else: 
                gridchoice=np.where(grid[1,:]==0)
                findx=[i for i in range(len(gridchoice[0]))]
                np.random.shuffle(findx)
                for f in findx:
                    if local_f(gridchoice[0][f])>0:                        
                        grid[1,gridchoice[0][f]]=1
                        counters[2]+=1
                        break
                    print('failed')

        # Update time and infection lists
        tsteps.append(dt+current_t)
        current_t=tsteps[-1]
        infecteds.append(count_type(2))
        exposeds.append(count_type(1))
        susceptibles.append(count_type(0))
        rowcounts=0
        for i in range(16):
            gridchoice=np.where(grid[i+1,:]==2)
            rowcounts=len(gridchoice[0])
            if rowcounts>3 and r4find[i]==0:
                r4[i]=(tsteps[-1])
                r4find[i]=1
            if rowcounts>10 and r11find[i]==0:
                r11[i]=(tsteps[-1])
                r11find[i]=1
            if rowcounts>7 and rowfind[i]==0:
                rows[i]=(tsteps[-1])
                rowfind[i]=1
        if infecteds[-1] == tot-15 or infecteds[-1]+exposeds[-1]==0:
            break

    inf_prop=[inf/240 for inf in infecteds]
    ax.plot(tsteps,inf_prop,'r:',alpha=0.3)
    return(rows,r4,r11)

rrs=np.zeros((16,100))   
r4s=np.zeros((16,100)) 
r1s=np.zeros((16,100)) 
fig, ax = plt.subplots(1,1,figsize=(4,4))
count=0
objuseI=np.array(objstoreI)
objuseII=np.array(objstoreII)
choicelist=[]

print("Running 100 posterior stochastic checks")

for i in range(len(objuseI)):
    if objuseI[i]<thresh and objuseII[i]<thresh:
        choicelist.append(i)
        
while count<100:  

    if count%10==0:
        print(count,'/100')
    choice=np.random.choice(choicelist)

    L=LstoreI[choice]
    ETA=BstoreI[choice]/(15*17)
    GAMMA=GstoreI[choice]
    DELTA=DstoreI[choice]
    BETAF=BPstoreI[choice]/(15*17)
    [r,r4,r11]=run_stochastic(L,ETA,GAMMA,DELTA,BETAF)
    # Used to find median, 1/4 and 3/4 arrival time - not plotted here
    rrs[:,count]=r   
    r4s[:,count]=r4
    r1s[:,count]=r11
    count+=1
    
ax.scatter(data_t,data_i)
ax.set_xlabel('Months')
ax.set_ylabel('Proportion Infected')
ax.set_xlim([0,85])

plt.tight_layout()

