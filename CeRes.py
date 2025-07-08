#
# class to measure the resolution of high-energy electrons
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import SurfaceIds as SID
import MyHist
import h5py
from scipy.stats import crystalball

def fxn_CrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(x,*pars)

def TargetFoil(tgtz):
    tgtz0 = -4300. # target center in detector coordinates
    tgtdz = 22.222222 # target spacing
    ntgt = 37 # number of target foils
    tgt0z = tgtz0 - 0.5*(ntgt-1)*tgtdz
    tgtnum = (tgtz-tgt0z)/tgtdz
    itgt = int(round(tgtnum))
    return itgt


class CeRes(object):
    def __init__(self,momrange,costrange,minNHits,minFitCon,minTrkQual):
        self.MomRange = momrange
        self.CosTRange = costrange
        self.minNHits = minNHits
        self.minFitCon = minFitCon
        self.minTrkQual = minTrkQual

        nDeltaMomBins = 200
        nMomBins = 200
        momrange=(self.MomRange[0],107)
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        momtitle="Momentum at "
        momxlabel="Momentum (MeV)"

        self.TrkLoc = [None]*3
        self.HOriginMom = MyHist.MyHist(name="OriginMom",label="MC Origin",bins=nMomBins, range=momrange,title="Momentum at Origin",xlabel="Momentum (MeV)")

        self.HTrkFitMom = [None]*3
        self.HTrkMCMom = [None]*3
        self.HTrkRespMom = [None]*3
        self.HTrkResoMom = [None]*3
        self.HTrkRefRespMom = [None]*3
        self.HTrkNotRefRespMom = [None]*3
        self.TrackerSIDs = [SID.TT_Front(), SID.TT_Mid(), SID.TT_Back()]
        momxlabel = "Momentum (MeV)"
        momresotitle = "Momentum Resolution"
        momresotitle = "Momentum Resolution"
        momresptitle = "Momentum Response"
        dmomxlabel = "$\\Delta$ Momentum (MeV)"
        # momentum in tracker
        for isid in range(len(self.TrackerSIDs)):
            loc = "@"+SID.SurfaceName(self.TrackerSIDs[isid])
            self.HTrkFitMom[isid] = MyHist.MyHist(name=loc+"Mom",label="Fit",bins=nMomBins, range=momrange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkMCMom[isid] = MyHist.MyHist(name=loc+"Mom",label="MC",bins=nMomBins, range=momrange,title="Momentum"+loc,xlabel=momxlabel)
            self.HTrkResoMom[isid] = MyHist.MyHist(name=loc+"Resolution",label="",bins=nDeltaMomBins, range=momresorange,title=momresotitle+loc,xlabel=dmomxlabel)
            self.HTrkRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="All",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkRefRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="Reflectable",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)
            self.HTrkNotRefRespMom[isid] = MyHist.MyHist(name=loc+"Response",label="Not Reflectable",bins=nDeltaMomBins, range=momresprange,title=momresptitle+loc,xlabel=dmomxlabel)

        # target intersections
        # momentum at target intersections

        tgtmomresptitle = "Target Momentum Response"
        self.HTgtAvgResp = MyHist.MyHist(name="AvgTgtResponse",label="Average",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespRef = MyHist.MyHist(name="AvgTgtResponseRef",label="Average (Reflectable)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtAvgRespNotRef = MyHist.MyHist(name="AvgTgtResponseNotRef",label="Average (Not Reflectable)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        self.HTgtLatestResp = MyHist.MyHist(name="LatestTgtResponse",label="Latest",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespRef = MyHist.MyHist(name="LatestTgtResponseRef",label="Latest (Reflectable)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)
        self.HTgtLatestRespNotRef = MyHist.MyHist(name="LatestTgtResponseNotRef",label="Latest (Not Reflectable)",bins=nDeltaMomBins, range=momresprange,title=tgtmomresptitle,xlabel=dmomxlabel)

        rhorange = [20,80]
        rhotitle ="Target Rho"
        rhoxlabel = "Rho (mm)"
        self.HTgtRho = MyHist.MyHist(name="HTgtRho",bins=100,range=rhorange,label="Fit",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoRef = MyHist.MyHist(name="HTgtRhoRef",bins=100,range=rhorange,label="Fit (Reflectable)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoNotRef = MyHist.MyHist(name="HTgtRhoNotRef",bins=100,range=rhorange,label="Fit (Not Reflectable)",title=rhotitle,xlabel=rhoxlabel)
        self.HTgtRhoMC = MyHist.MyHist(name="HTgtRho",bins=100,range=rhorange,label="MC",title=rhotitle,xlabel=rhoxlabel)
        self.HOriginRho = MyHist.MyHist(name="HTgtRho",bins=100,range=rhorange,label="MC Origin",title=rhotitle,xlabel=rhoxlabel)
        foilrange = [-0.5,36.5]
        foiltitle ="Target Foil"
        foilxlabel="Foil #"
        self.HTgtFoil = MyHist.MyHist(name="HTgtFoil",bins=37,range=foilrange,label="Fit",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilRef = MyHist.MyHist(name="HTgtFoilRef",bins=37,range=foilrange,label="Fit (Reflectable)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilNotRef = MyHist.MyHist(name="HTgtFoilNotRef",bins=37,range=foilrange,label="Fit (Not Reflectable)",title=foiltitle,xlabel=foilxlabel)
        self.HTgtFoilMC = MyHist.MyHist(name="HTgtFoil",bins=37,range=foilrange,label="MC",title=foiltitle,xlabel=foilxlabel)
        self.HOriginFoil = MyHist.MyHist(name="HTgtFoil",bins=37,range=foilrange,label="MC Origin",title=foiltitle,xlabel=foilxlabel)
        costrange = [-0.8,0.8]
        costtitle ="Target Cos($\\Theta$)"
        costxlabel="Cos($\\Theta$)"
        self.HTgtCosT = MyHist.MyHist(name="HTgtCosT",bins=100,range=costrange,label="Fit",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTRef = MyHist.MyHist(name="HTgtCosT",bins=100,range=costrange,label="Fit (Reflectable)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTNotRef = MyHist.MyHist(name="HTgtCosT",bins=100,range=costrange,label="Fit (Not Reflectable)",title=costtitle,xlabel=costxlabel)
        self.HTgtCosTMC = MyHist.MyHist(name="HTgtCosT",bins=100,range=costrange,label="MC",title=costtitle,xlabel=costxlabel)
        self.HOriginCosT = MyHist.MyHist(name="HTgtCosT",bins=100,range=costrange,label="MC Origin",title=costtitle,xlabel=costxlabel)

        self.HTrkQual = MyHist.MyHist(name="HTrkQual",bins=100,range=[0.0,1.0],label="TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HFitCon = MyHist.MyHist(name="HFitCon",bins=100,range=[0.0,1.0],label="FitCon",title="Fit Consistency",xlabel="")
        self.HNHits = MyHist.MyHist(name="HNHits",bins=100,range=[0.5,100.5],label="NActive",title="Fit N Hits",xlabel="N Hits")

    def Loop(self,files):
        elPDG = 11
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk|trksegs|trkmcsim|trksegsmc|trkqual/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkQual = batch['trkqual.result']  # track fit quality
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            # should be 1 track/event
            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            Segs = segs[:,0]
            FitCon = fitcon[:,0]
            NHits = nhits[:,0]
            TrkQual = trkQual[:,0]
            assert(len(Segs)==len(NHits))

            self.HTrkQual.fill(np.array(TrkQual))
            self.HFitCon.fill(np.array(FitCon))
            self.HNHits.fill(np.array(NHits))

            # define good MC selection first, to allow downstream comparisons
            SegsMC = segsMC[:,0] # segments (of 1st MC match) of 1st track
            TrkMC = trkMC[:,0,0] # primary MC match of 1st track
            # basic consistency test
            assert((len(runnum) == len( Segs)) & (len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(NHits) == len(Segs)))
            goodMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
            OMom = TrkMC[goodMC].mom.magnitude()
            goodMC = goodMC & (OMom>self.MomRange[0]) & (OMom < self.MomRange[1])
            OMom = OMom[goodMC]
            self.HOriginMom.fill(np.array(OMom))
            self.HOriginRho.fill(np.array(TrkMC[goodMC].pos.rho()))
            self.HOriginCosT.fill(np.array(TrkMC[goodMC].mom.cosTheta()))
            self.HOriginFoil.fill(np.array(list(map(TargetFoil,TrkMC[goodMC].pos.z()))))

            # truncate accordingly
            SegsMC = SegsMC[goodMC]
            Segs = Segs[goodMC]
            NHits = NHits[goodMC]
            FitCon = FitCon[goodMC]
            TrkQual = TrkQual[goodMC]
            midsegs = Segs[(Segs.sid == SID.TT_Mid()) & (Segs.mom.z() > 0.0) ]
            CosT = ak.flatten(midsegs.mom.cosTheta())
            goodFit = (NHits >= self.minNHits) & (FitCon > self.minFitCon) & (TrkQual > self.minTrkQual) & (CosT > self.CosTRange[0]) & (CosT < self.CosTRange[1])
            TSDASeg = Segs[Segs.sid == SID.TSDA() ]
            noTSDA = ak.num(TSDASeg)==0


            # sample the fits at the specified
            for isid in range(len(self.TrackerSIDs)) :
                sid = self.TrackerSIDs[isid]
                segs = Segs[(Segs.sid == sid) & (Segs.mom.z() > 0.0) ]
                assert(len(segs) == len(Segs))
                mom = segs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
                hasmom = ak.count_nonzero(mom,axis=1)==1
                segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
                momMC = segsMC.mom.magnitude()
                hasMC = ak.count_nonzero(momMC,axis=1)==1
                good = hasMC & goodFit & hasmom
                reflectable = good & noTSDA
                notreflectable = good & np.logical_not(noTSDA)
                goodmom = mom[good]
                goodmom = ak.flatten(goodmom,axis=1)
                refmom = mom[reflectable]
                refmom = ak.flatten(refmom,axis=1)
                notrefmom = mom[notreflectable]
                notrefmom = ak.flatten(notrefmom,axis=1)
                goodmomMC = momMC[good]
                goodmomMC = ak.flatten(goodmomMC,axis=1)
                assert(len(goodmom) == len(goodmomMC) )
                self.HTrkFitMom[isid].fill(np.array(goodmom))
                self.HTrkMCMom[isid].fill(np.array(goodmomMC))
                momreso = goodmom - goodmomMC
                self.HTrkResoMom[isid].fill(np.array(momreso))
                momresp = goodmom - OMom[good]
                self.HTrkRespMom[isid].fill(np.array(momresp))
                momrefresp = refmom - OMom[reflectable]
                momnotrefresp = notrefmom - OMom[notreflectable]
                self.HTrkRefRespMom[isid].fill(np.array(momrefresp))
                self.HTrkNotRefRespMom[isid].fill(np.array(momnotrefresp))
            #foil response
            reflectable = noTSDA
            notreflectable = np.logical_not(noTSDA)
            tgtsegs = Segs[(Segs.sid == SID.ST_Foils()) & goodFit]
            tgtsegsref = tgtsegs[reflectable]
            tgtsegsnotref = tgtsegs[notreflectable]
            tgtmom = tgtsegs.mom.magnitude()
            tgtmomref = tgtsegsref.mom.magnitude()
            tgtmomnotref = tgtsegsnotref.mom.magnitude()
            tgtrho = tgtsegs.pos.rho()
            tgtrhoref = tgtsegsref.pos.rho()
            tgtrhonotref = tgtsegsnotref.pos.rho()
            tgtcost = tgtsegs.mom.cosTheta()
            tgtcostref = tgtsegsref.mom.cosTheta()
            tgtcostnotref = tgtsegsnotref.mom.cosTheta()
            tgtz = tgtsegs.pos.z()
            tgtzref = tgtsegsref.pos.z()
            tgtznotref = tgtsegsnotref.pos.z()
            ntgts = ak.count(tgtmom,axis=1)
            ntgtsref = ak.count(tgtmomref,axis=1)
            ntgtsnotref = ak.count(tgtmomnotref,axis=1)
            goodtgt = (ntgts > 0)
            goodtgtref = (ntgtsref > 0)
            goodtgtnotref = (ntgtsnotref > 0)
            ntgts = ntgts[goodtgt]
            ntgtsref = ntgtsref[goodtgtref]
            ntgtsnotref = ntgtsnotref[goodtgtnotref]
            avgmom = ak.sum(tgtmom,axis=1)
            avgmomref = ak.sum(tgtmomref,axis=1)
            avgmomnotref = ak.sum(tgtmomnotref,axis=1)
            avgmom = avgmom[goodtgt]/ntgts
            avgmomref = avgmomref[goodtgtref]/ntgtsref
            avgmomnotref = avgmomnotref[goodtgtnotref]/ntgtsnotref
            omomtgt = OMom[goodtgt]
            omomtgtref = OMom[goodtgtref]
            omomtgtnotref = OMom[goodtgtnotref]
            avgtgtresp = avgmom - omomtgt
            avgtgtrespref = avgmomref - omomtgtref
            avgtgtrespnotref = avgmomnotref - omomtgtnotref
            #
            self.HTgtAvgResp.fill(np.array(avgtgtresp))
            self.HTgtAvgRespRef.fill(np.array(avgtgtrespref))
            self.HTgtAvgRespNotRef.fill(np.array(avgtgtrespnotref))
            # find the latest target intersection
            tgtmomlate = tgtsegs[ak.argsort(tgtsegs.time,ascending=False)].mom.magnitude()
            tgtmomlate = ak.flatten(tgtmomlate[:,:1])
            tgtmomlateref = tgtsegsref[ak.argsort(tgtsegsref.time,ascending=False)].mom.magnitude()
            tgtmomlateref = ak.flatten(tgtmomlateref[:,:1])
            tgtmomlatenotref = tgtsegsnotref[ak.argsort(tgtsegsnotref.time,ascending=False)].mom.magnitude()
            tgtmomlatenotref = ak.flatten(tgtmomlatenotref[:,:1])

            latetgtresp = tgtmomlate - omomtgt
            latetgtrespref = tgtmomlateref - omomtgtref
            latetgtrespnotref = tgtmomlatenotref - omomtgtnotref
            self.HTgtLatestResp.fill(np.array(latetgtresp))
            self.HTgtLatestRespRef.fill(np.array(latetgtrespref))
            self.HTgtLatestRespNotRef.fill(np.array(latetgtrespnotref))

            self.HTgtRho.fill(np.array(ak.flatten(tgtrho)))
            self.HTgtRhoRef.fill(np.array(ak.flatten(tgtrhoref)))
            self.HTgtRhoNotRef.fill(np.array(ak.flatten(tgtrhonotref)))
            self.HTgtCosT.fill(np.array(ak.flatten(tgtcost)))
            self.HTgtCosTRef.fill(np.array(ak.flatten(tgtcostref)))
            self.HTgtCosTNotRef.fill(np.array(ak.flatten(tgtcostnotref)))
            self.HTgtFoil.fill(np.array(list(map(TargetFoil,ak.flatten(tgtz)))))
            self.HTgtFoilRef.fill(np.array(list(map(TargetFoil,ak.flatten(tgtzref)))))
            self.HTgtFoilNotRef.fill(np.array(list(map(TargetFoil,ak.flatten(tgtznotref)))))


            tgtsegsmc = SegsMC[(SegsMC.sid == SID.ST_Foils())]
            self.HTgtRhoMC.fill(np.array(ak.flatten(tgtsegsmc.pos.rho())))
            self.HTgtCosTMC.fill(np.array(ak.flatten(tgtsegsmc.mom.cosTheta())))
            self.HTgtFoilMC.fill(np.array(list(map(TargetFoil,ak.flatten(tgtsegsmc.pos.z())))))

            # test for missing intersections
            hasent = (Segs.sid == 0) & (Segs.mom.z() > 0.0)
            hasmid = (Segs.sid == 1) & (Segs.mom.z() > 0.0)
            hasxit = (Segs.sid == 2) & (Segs.mom.z() > 0.0)
            hasent = ak.any(hasent,axis=1)
            hasmid = ak.any(hasmid,axis=1)
            hasxit = ak.any(hasxit,axis=1)
            hasall = hasent & hasmid & hasxit
            missing = ak.count(hasall,0) - ak.count_nonzero(hasall)
            if(missing > 0):
                print("Found",missing,"Instances of missing intersections in",ak.count(hasall,0),"tracks")
                for itrk in range(len(hasall)):
                    if (not hasall[itrk]):
                        print("Missing intersection: ",hasent[itrk],hasmid[itrk],hasxit[itrk]," eid ",runnum[itrk],":",subrun[itrk],":",event[itrk],sep="")
        print()


    def PlotQuality(self):
        fig, (atrkqual,anhits,afitcon) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        self.HTrkQual.plot(atrkqual)
        self.HNHits.plot(anhits)
        self.HFitCon.plot(afitcon)

    def PlotTrackerMomentum(self):
        fig, (amom,areso,aresp) = plt.subplots(3,3,layout='constrained', figsize=(15,15))
        for isid in range(len(self.TrackerSIDs)) :
            self.HTrkFitMom[isid].plot(amom[isid])
            self.HTrkMCMom[isid].plot(amom[isid])
            self.HTrkResoMom[isid].plot(areso[isid])
            self.HTrkRespMom[isid].plot(aresp[isid])
            self.HTrkRefRespMom[isid].plot(aresp[isid])
            self.HTrkNotRefRespMom[isid].plot(aresp[isid])
        amom[0].legend(loc="upper left")
        aresp[0].legend(loc="upper left")

    def PlotTarget(self):
        fig, ((arho,afoil,acost),(avgresp,latestresp,nnresp)) = plt.subplots(2,3,layout='constrained', figsize=(15,10))
        # Rho
        self.HTgtRho.plot(arho)
        self.HTgtRhoRef.plot(arho)
        self.HTgtRhoNotRef.plot(arho)
        self.HTgtRhoMC.plot(arho)
        self.HOriginRho.plot(arho)
        arho.legend(loc="upper left")
        # Foil
        self.HTgtFoil.plot(afoil)
        self.HTgtFoilRef.plot(afoil)
        self.HTgtFoilNotRef.plot(afoil)
        self.HTgtFoilMC.plot(afoil)
        self.HOriginFoil.plot(afoil)
        afoil.legend(loc="upper left")
        # Cos(theta)
        self.HTgtCosT.plot(acost)
        self.HTgtCosTRef.plot(acost)
        self.HTgtCosTNotRef.plot(acost)
        self.HTgtCosTMC.plot(acost)
        self.HOriginCosT.plot(acost)
        acost.legend(loc="upper left")
        # Response: Average
        self.HTgtAvgResp.plot(avgresp)
        self.HTgtAvgRespRef.plot(avgresp)
        self.HTgtAvgRespNotRef.plot(avgresp)
        avgresp.legend(loc="upper left")
        # Latest Foil
        self.HTgtLatestResp.plot(latestresp)
        self.HTgtLatestRespRef.plot(latestresp)
        self.HTgtLatestRespNotRef.plot(latestresp)
        latestresp.legend(loc="upper left")

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            self.HOriginMom.save(hdf5file)
            for isid in range(len(self.TrackerSIDs)) :
                self.HTrkFitMom[isid].save(hdf5file)
                self.HTrkMCMom[isid].save(hdf5file)
                self.HTrkResoMom[isid].save(hdf5file)
                self.HTrkRespMom[isid].save(hdf5file)
                self.HTrkRefRespMom[isid].save(hdf5file)
                self.HTrkNotRefRespMom[isid].save(hdf5file)
            self.HTgtAvgResp.save(hdf5file)
            self.HTgtRho.save(hdf5file)
            self.HTgtRhoMC.save(hdf5file)
            self.HOriginRho.save(hdf5file)
            self.HTgtFoil.save(hdf5file)
            self.HTgtFoilMC.save(hdf5file)
            self.HOriginFoil.save(hdf5file)
            self.HTgtCosT.save(hdf5file)
            self.HTgtCosTMC.save(hdf5file)
            self.HOriginCosT.save(hdf5file)
