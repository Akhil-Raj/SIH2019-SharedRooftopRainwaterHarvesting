def YASRec(Vprev, Qcurr, Dcurr, A, R, Rcurr, S, nd, nft, F):
    vprev = Vprev / (A * R)
    qcurr = Qcurr / (A * R)
    dcurr = findDcurr(nd, nft, F) / (A * R)
    s = S / (A * R)

    ycurr = min(dcurr, vprev)
    ocurr = max(0, vprev + Rcurr / R - s)
    vcurr = vprev + qcurr - ycurr - ocurr


    return (vcurr, ycurr, ocurr)

def findDcurr(nd, nft, F):
    Dcurr = nd * nft * F

if __name__ == '__main__':
    YASRec()
    #findDcurr()