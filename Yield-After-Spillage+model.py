
# coding: utf-8

# In[139]:


import math

def findDcurr(nd, nft, F):
    Dcurr = nd * nft * F
    
def YASRec(Vprev, Qcurr, Dcurr, A, R, Rcurr, S, nd, nft, F):
    vprev = Vprev / (A * R)
    qcurr = Qcurr / (A * R)
    #dcurr = findDcurr(nd, nft, F) / (A * R)
    dcurr = nd * nft * F / (A * R)
    s = S / (A * R)
    ycurr = min(dcurr, vprev)
    ocurr = max(0, vprev + Rcurr / R - s)
    vcurr = vprev + qcurr - ycurr - ocurr
    return (vcurr, ycurr, ocurr)

def water_saving(Vprev, Qcurr, Dcurr, A, R, Rcurr, S, nd, nft, F) :
    Yt = []
    Dt = []
    yt = 20000
    Yn = Dn = 0
    for t in range(365):
        k = YASRec(yt, 60, 25, 69, 80, 40,64,83,96,57)
        yt = k[1]
        dt = nd * nft * F
        Yt.append(yt)
        Dt.append(dt)
        Yn = Yn + yt
        Dn = Dn + dt
        WSn = Yn/Dn
        s = S / (A * R)
        d = Dn / (A * R*365)
    print(WSn)

def water_save(x1, x2, x3, x4, s, d, f):
    frac = x1*s/(x2+s)
    WS = frac*pow(d,x3)*pow(f,x4)
    return(WS)

def Cost_Anal(a, b, k, S, D, r, N, Ci, Cw, Ce):
    WS = water_save(45.547, 3.052, -0.492, -0.264, 40 , 2, 0.9)
    #capital cost
    Cs = a + b*S #equipment cost
    Cc = Cs+Ci # Ci-> installation cost
    #operational cost
    Cywm = Cw*(1-WS/100)*D*365  #Cw->unit cost of drinking water
    Cye=Ce*(WS)*D*3.65    #Ce->cost of energy(pm3)
    #maintenance cost
    Cm = k*Cc
    #total cost
    frac = (pow(1+r, N)-1)/r*pow(1+r,N)
    PV = Cc + frac*(Cywm + Cye + Cm)
    return(Cc, Cm, Cye, Cywm)
    
def optimization(b, k, d, f, r, x1, x2, x3, x4, A, R, N, Cw, Ce, D):
    cost = Cost_Anal(0, 82, 0.02, 3, 0.575, 0.03, 5, 200, 4.17, 0.02)
    Cc = cost[0]
    Cm = cost[1]
    Cye = cost[2]
    Cywm = cost[3]
    num = (pow(1+r, N)-1)*x1*x2*365*(Cw-Ce) 
    den1 = 100*b
    den2 = den1*((pow((1+r),N))*(r+k)-k) 
    p1 = pow(d, 0.5*(1+x3)) 
    p2 = pow(d, 0.5*x4)
    #optimal tank size 
    Sopt = A*R*math.sqrt(num/den2)*p1*p2-x2
    #payback period of investment
    T = Cc/(Cw*D*365*Cm*Cye*Cywm)
    return(Sopt, T)
if __name__ == '__main__': 
    opt = optimization(82, 0.03, 0.2,0.5, 0.03 ,45.547, 3.052, -0.492, -0.264,160,9.5,25, 4.17, 0.02, .0575)
    print(opt)

