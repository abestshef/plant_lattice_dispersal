
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

# CODE TO RUN THE DISPERSAL MODEL
REPS = int(input("How many replicates do you wish to run? (roughly 20 runs per minute on a laptop)"))
THRESH=float(input("What threshold do you wish to use? (should be less than 1 - more replicates allow a lower threshold)"))

# THE STOCHASTIC SIMULAITON CODE

def run_stochastic(ALPHA, BETA, GAMMA, DELTA, ALPHAP, BETAP,pl):
    
    # Function to count noumbers in each compartment
    def count_type(sirtype):
        return len(np.where(grid==sirtype)[0])

    # Function to find current scale for next time-step
    def findscale(phi,lastgrid,ttype):
        # To save time, we only calculate the change to phi from last transition
        # lastgrid is co-ords of last cell to change
        # ttype is what it changed to
        
        E=count_type(1)

        
        if ttype==1:
        # get rid of the contributions from that S
            gridi=np.where(grid==2)
            for i in range(len(gridi[0])):
                d=np.sqrt((gridi[0][i]-lastgrid[0])**2+(gridi[1][i]-lastgrid[1])**2)
                if gridi[0][i]>0:
                    phi-=BETA*np.exp(-d/ALPHA)/(2*np.pi*ALPHA**2)
                else:

                    phi-=BETAP*np.exp(-d/ALPHAP)/(2*np.pi*ALPHAP**2)
        elif ttype==2:
            # add on new infections from the I
            grids=np.where(grid==0)
            for j in range(len(grids[0])):
                d=np.sqrt((lastgrid[0]-grids[0][j])**2+(lastgrid[1]-grids[1][j])**2)
                phi+=BETA*np.exp(-d/ALPHA)/(2*np.pi*ALPHA**2)

        scale=GAMMA*E+phi
                  
        return scale,phi

    def local_inf(site_x,site_y):
        localinfs=0

        if site_x>1:
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

    grid=np.zeros((17,15))
    grid[0,:]=2
    tsteps=[0]
    infecteds=[0]
    infpairs=[0]
    exposeds=[0]
    current_t=DELTA # Only start simulation after DELTA delay
    
    timepoint=0
    obj=0
    objpairs=0
    
    # Need to do a first calculation of phi before run
    lastgrid=[0,0] # Although not used first time, needs a value
    ttype=0 # Although not used first time, needs a value

    phi=0

    gridi=np.where(grid==2)
    grids=np.where(grid==0)

    for i in range(15):
        for j in range(len(grids[0])):
            d=np.sqrt((0-grids[0][j])**2+(i-grids[1][j])**2)
            phi+=BETAP*np.exp(-d/ALPHAP)/(2*np.pi*ALPHAP**2)

  # Main run
    
    while current_t<90:

        # Find time-step
        scale,phi=findscale(phi,lastgrid,ttype)
        dt = -np.log(np.random.rand()) / scale
        #if int(current_t+dt)>int(current_t):
            #print(current_t, susceptibles[-1])
        
        #Choose event
        if np.random.rand()<GAMMA*exposeds[-1]/scale: #Event is E -> I

            gridchoice=np.where(grid==1)
            fr=np.random.randint(0,len(gridchoice[0]))
            grid[gridchoice[0][fr],gridchoice[1][fr]]=2
            lastgrid=[gridchoice[0][fr],gridchoice[1][fr]]
            ttype=2
            # add on pairs
            newpairs=local_inf(gridchoice[0][fr],gridchoice[1][fr])
            infpairs.append(infpairs[-1]+newpairs)
        else: #Event is S -> E   
            phi_now=0
            found=0
            gridi=np.where(grid==2)
            grids=np.where(grid==0)
            check=np.random.rand()
    
            for i in range(len(gridi[0])):
                if found==1:
                    break
                for j in range(len(grids[0])):
                    d=np.sqrt((gridi[0][i]-grids[0][j])**2+(gridi[1][i]-grids[1][j])**2)
                    if gridi[0][i]>0:
                        phi_now+=BETA*np.exp(-d/ALPHA)/(2*np.pi*ALPHA**2)
                    else:
                        phi_now+=BETAP*np.exp(-d/ALPHAP)/(2*np.pi*ALPHAP**2)
                    if phi_now/phi > check:
                        grid[grids[0][j],grids[1][j]]=1
                        found=1
                        lastgrid=[grids[0][j],grids[1][j]]
                        ttype=1
                    
                        break
                

        # Update time and infection lists
        tsteps.append(dt+current_t)
        current_t=tsteps[-1]
        infecteds.append(count_type(2)-15)
        exposeds.append(count_type(1))
        
        if tsteps[-1]>data_t[timepoint]:
            inf_check=infecteds[-1]/(tot-15)
            obj+=(inf_check-data_cum[timepoint])**2
            objpairs+=(infpairs[-1]/(16*14+15*15)-data_ii[timepoint])**2
            timepoint+=1
            if obj>1 or timepoint==14:
                break
        
        if infecteds[-1] == tot-15 or infecteds[-1]+exposeds[-1]==0:
            break
  
    # Occasionally a large early time-step ends the sim but reports
    # an obj<0.5 and these need to be re-evaluated
    if timepoint<14:
        for topup in range(timepoint,14):
            obj+=(infecteds[-1]/(tot-15)-data_cum[topup])**2
    
    if pl==True:
        inf_prop=[inf/240 for inf in infecteds]
        ax.plot(tsteps,inf_prop,'r:',alpha=0.3)
    return([obj,objpairs])

#Store for the posterior parameters
LstoreI=[]
BstoreI=[]
GstoreI=[]
DstoreI=[]
BPstoreI=[]
objstoreI=[]
objstoreII=[]

for reps in range(REPS):
    if reps % 10 == 0:
        print(reps, 'simulations complete')
    tot=17*15
    ALPHA=10*np.random.rand()
    ETA=2*np.random.rand()
    ALPHAP=ALPHA
    ETAP=2*np.random.rand()
    GS=0.025+19.975*np.random.rand()
    GAMMA=1/GS
    DELTA=30*np.random.rand()
    pl=False
    infchecks=run_stochastic(ALPHA,ETA,GAMMA,DELTA,ALPHAP,ETAP,pl)

    if infchecks[0]<1 and infchecks[1]<1:
        LstoreI.append(ALPHA)
        BstoreI.append(ETA)
        GstoreI.append(GAMMA)
        DstoreI.append(DELTA)
        #ALPstoreI.append(ALPHAP)
        BPstoreI.append(ETAP)
        objstoreI.append(infchecks[0])
        objstoreII.append(infchecks[1])

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
ax[0].hist(Ls, range=(0,10),bins=20, label="$L$")
ax[0].set_ylabel('Count')
ax[0].set_xlabel(r'Local ratio, $\alpha$')

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

# RUN STOCHASTIC POSTERIOR CHECKS

print("Running 100 posterior stochastic checks")

for i in range(len(objstoreI)):
    if objstoreI[i]<thresh and objstoreII[i]<thresh:
        choicelist.append(i)
fig, ax = plt.subplots(1,1,figsize=(4,4))
count=0    
while count<100:  

    if count%10==0:
        print(count,'/100')
    choice=np.random.choice(choicelist)

    L=LstoreI[choice]
    ETA=BstoreI[choice]
    GAMMA=GstoreI[choice]
    DELTA=DstoreI[choice]
    BETAF=BPstoreI[choice]
    pl=True
    run_stochastic(L,ETA,GAMMA,DELTA,L,BETAF,pl)
    count+=1

ax.scatter(data_t,data_i)
ax.set_xlabel('Months')
ax.set_ylabel('Proportion Infected')
ax.set_xlim([0,85])

plt.tight_layout()



