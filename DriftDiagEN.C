#include "TCut.h"
#include "TTree.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include "THStack.h"
#include "TStyle.h"
#include "TLegend.h"
#include <fstream>
#include "FillChain.C"
#include "KKCutsEN.hh"
void DriftDiag(TTree* ta,const char* savesuffix="") {
  gStyle->SetStatH(0.2);
  gStyle->SetStatW(0.2);
  gStyle->SetOptFit(111);

  const int NBINS(100);

  TH2D* dd = new TH2D("dd","Reco DOCA vs MC true DOCA;MC DOCA (mm);Reco DOCA (mm)",50,-3,3,50,-3,3);
  TH1D* fcon = new TH1D("fcon","Fit Consistency",NBINS,0.0,1.0);
  TH1D* entmomres = new TH1D("entmomres","Momentum Resolution at Tracker Entrance;#Delta mom (MeV/c)",NBINS,-5,5);
  TH1D* xitmomres = new TH1D("xitmomres","Momentum Resolution at Tracker Exit;#Delta mom (MeV/c)",NBINS,-5,5);
  TH1D* midmomres = new TH1D("midmomres","Momentum Resolution at Tracker Middle;#Delta mom (MeV/c)",NBINS,-5,5);
  TH1D* midmompull = new TH1D("midmompull","Momentum Pull at Tracker Middle;#Delta midmom/#sigma_{mom}",NBINS,-10,10);
  TH1D* t0res = new TH1D("t0res","T0 Resolution;#Delta T0 (ns)",NBINS,-10,10);
  TH1D* t0pull = new TH1D("t0pull","T0 Pull;#Delta T0/#sigma_{t0}",NBINS,-10,10);
  TH1D* dres = new TH1D("dres","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
  TH1D* dresg = new TH1D("dresg","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
  TH1D* dresb = new TH1D("dresb","Drift Resolution;R_{drift}-MC DOCA (mm)",NBINS,-2.5,2.5);
  TH1D* dresp = new TH1D("dresp","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
  TH1D* drespg = new TH1D("drespg","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
  TH1D* drespb = new TH1D("drespb","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}",NBINS,-8.0,8.0);
  TH1D* udoca = new TH1D("udoca","Unbiased DOCA Resolution;UDOCA-MC DOCA) (mm)",NBINS,-5,5);
  TH1D* udocap = new TH1D("udocap","Unbiased DOCA Resolution Pull;UDOCA-MC DOCA/UDOCA error",NBINS,-15,15);

  TH1D* amcdoca = new TH1D("amcdoca","DOCA, LR Hits;RDrift (mm)",NBINS,-0.2,3.5);
  TH1D* tmcdoca = new TH1D("tmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* uamcdoca = new TH1D("uamcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* gmcdoca = new TH1D("gmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* dmcdoca = new TH1D("dmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* bmcdoca = new TH1D("bmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* ngmcdoca = new TH1D("ngmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* nbmcdoca = new TH1D("nbmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* icmcdoca = new TH1D("icmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* ibmcdoca = new TH1D("ibmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* fmcdoca = new TH1D("fmcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* umcdoca = new TH1D("umcdoca","MC DOCA",NBINS,-0.2,3.5);
  TH1D* rmcdoca = new TH1D("rmcdoca","MC DOCA",NBINS,-0.2,3.5);

  TH1D* gcdrift = new TH1D("gcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* dcdrift = new TH1D("dcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* bcdrift = new TH1D("bcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* ngcdrift = new TH1D("ngcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* nbcdrift = new TH1D("nbcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* iccdrift = new TH1D("iccdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* ibcdrift = new TH1D("ibcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* fcdrift = new TH1D("fcdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* ucdrift = new TH1D("ucdrift","CDrift",NBINS,-0.2,3.5);
  TH1D* rcdrift = new TH1D("rcdrift","CDrift",NBINS,-0.2,3.5);

  TH1D* ardrift = new TH1D("ardrift","DOCA, LR Hits; RDrift (mm)",NBINS,-0.2,3.5);
  TH1D* grdrift = new TH1D("grdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* drdrift = new TH1D("drdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* brdrift = new TH1D("brdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* ngrdrift = new TH1D("ngrdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* nbrdrift = new TH1D("nbrdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* icrdrift = new TH1D("icrdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* ibrdrift = new TH1D("ibrdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* frdrift = new TH1D("frdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* urdrift = new TH1D("urdrift","RDrift",NBINS,-0.2,3.5);
  TH1D* rrdrift = new TH1D("rrdrift","RDrift",NBINS,-0.2,3.5);

  TH1D* audoca = new TH1D("audoca","DOCA, LR Hits; RDrift (mm)",NBINS,-0.2,3.5);
  TH1D* gudoca = new TH1D("gudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* dudoca = new TH1D("dudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* budoca = new TH1D("budoca","UDOCA",NBINS,0.0,3.5);
  TH1D* ngudoca = new TH1D("ngudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* nbudoca = new TH1D("nbudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* icudoca = new TH1D("icudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* ibudoca = new TH1D("ibudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* fudoca = new TH1D("fudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* uudoca = new TH1D("uudoca","UDOCA",NBINS,0.0,3.5);
  TH1D* rudoca = new TH1D("rudoca","UDOCA",NBINS,0.0,3.5);

  TH1D* bbqual = new TH1D("bbqual","Bkg Quality;Quality Value",110,-0.05,1.05);
  TH1D* cbqual = new TH1D("cbqual","Bkg Quality;Quality Value",110,-0.05,1.05);

  TH1D* psqual = new TH1D("psqual","Correct/Incorrect Sign Proability vs Sign Quality;Quality Value;Sign Probability Ratio",110,-0.05,1.05);
  TH1D* gsqual = new TH1D("gsqual","Sign Quality;Quality Value",110,-0.05,1.05);
  TH1D* bsqual = new TH1D("bsqual","Sign Quality;Quality Value",110,-0.05,1.05);

  TH1D* gdqual = new TH1D("gdqual","Drift Quality;Quality Value",110,-0.05,1.05);
  TH1D* bdqual = new TH1D("bdqual","Drift Quality;Quality Value",110,-0.05,1.05);

  TH1D* calotres = new TH1D("calotres","Calo Hit Time Residual ;#Delta t (ns)",100,-5,5);
  TH1D* calotpull = new TH1D("calotpull","Calo Hit Time Pull;#Delta t (ns)/#sigma_{#Delta t}",100,-5,5);

  fcon->SetStats(0);
  dd->SetStats(0);
  ta->Project("dd","trkhits[].udoca:trkhitsmc[].doca",gfit+ghit+thit);
  ta->Project("amcdoca","trkhitsmc[].dist",gfit+dhit);
  ta->Project("uamcdoca","trkhitsmc[].dist",gfit+ahit+thit);
  ta->Project("tmcdoca","trkhitsmc[].dist",gfit+thit);
  ta->Project("gmcdoca","trkhitsmc[].dist",gfit+dhit+gambig+gdrift);
  ta->Project("dmcdoca","trkhitsmc[].dist",gfit+dhit+gambig+bdrift);
  ta->Project("bmcdoca","trkhitsmc[].dist",gfit+dhit+bambig);
  ta->Project("ngmcdoca","trkhitsmc[].dist",gfit+nhit+gdrift);
  ta->Project("nbmcdoca","trkhitsmc[].dist",gfit+nhit+bdrift);
  ta->Project("icmcdoca","trkhitsmc[].dist",gfit+ihit+thit);
  ta->Project("ibmcdoca","trkhitsmc[].dist",gfit+ihit+!thit);
  ta->Project("umcdoca","trkhitsmc[].dist",gfit+uhit+ghit);
  ta->Project("rmcdoca","trkhitsmc[].dist",gfit+rhit+ghit);

  ta->Project("gcdrift","trkhits[].cdrift",gfit+dhit+gambig+gdrift);
  ta->Project("dcdrift","trkhits[].cdrift",gfit+dhit+gambig+bdrift);
  ta->Project("bcdrift","trkhits[].cdrift",gfit+dhit+bambig);
  ta->Project("ngcdrift","trkhits[].cdrift",gfit+nhit+gdrift);
  ta->Project("nbcdrift","trkhits[].cdrift",gfit+nhit+bdrift);
  ta->Project("iccdrift","trkhits[].cdrift",gfit+ihit+thit);
  ta->Project("ibcdrift","trkhits[].cdrift",gfit+ihit+!thit);
  ta->Project("ucdrift","trkhits[].cdrift",gfit+uhit+ghit);
  ta->Project("rcdrift","trkhits[].cdrift",gfit+rhit+ghit);

  ta->Project("ardrift","trkhits[].rdrift",gfit+dhit);
  ta->Project("grdrift","trkhits[].rdrift",gfit+dhit+gambig+gdrift);
  ta->Project("drdrift","trkhits[].rdrift",gfit+dhit+gambig+bdrift);
  ta->Project("brdrift","trkhits[].rdrift",gfit+dhit+bambig);
  ta->Project("ngrdrift","trkhits[].rdrift",gfit+nhit+gdrift);
  ta->Project("nbrdrift","trkhits[].rdrift",gfit+nhit+bdrift);
  ta->Project("icrdrift","trkhits[].rdrift",gfit+ihit+thit);
  ta->Project("ibrdrift","trkhits[].rdrift",gfit+ihit+!thit);
  ta->Project("urdrift","trkhits[].rdrift",gfit+uhit+ghit);
  ta->Project("rrdrift","trkhits[].rdrift",gfit+rhit+ghit);

  ta->Project("audoca","abs(trkhits[].udoca)",gfit+dhit);
  ta->Project("gudoca","abs(trkhits[].udoca)",gfit+dhit+gambig+gdrift);
  ta->Project("dudoca","abs(trkhits[].udoca)",gfit+dhit+gambig+bdrift);
  ta->Project("budoca","abs(trkhits[].udoca)",gfit+dhit+bambig);
  ta->Project("ngudoca","abs(trkhits[].udoca)",gfit+nhit+gdrift);
  ta->Project("nbudoca","abs(trkhits[].udoca)",gfit+nhit+bdrift);
  ta->Project("icudoca","abs(trkhits[].udoca)",gfit+ihit+thit);
  ta->Project("ibudoca","abs(trkhits[].udoca)",gfit+ihit+!thit);
  ta->Project("uudoca","abs(trkhits[].udoca)",gfit+uhit+ghit);
  ta->Project("rudoca","abs(trkhits[].udoca)",gfit+rhit+ghit);

  ta->Project("fcon","trk.fitcon",mcpri+gfit);
  ta->Project("entmomres","trksegs[][trkmcvd.iinter].mom.R()-trkmcvd.mom.R()",mcent+gfit);
  ta->Project("xitmomres","trksegs[][trkmcvd.iinter].mom.R()-trkmcvd.mom.R()",mcxit+gfit);
  ta->Project("midmomres","trksegs[][trkmcvd.iinter].mom.R()-trkmcvd.mom.R()",mcmid+gfit);
  ta->Project("midmompull","(1.0/trksegs[][trkmcvd.iinter].momerr)*(trksegs[][trkmcvd.iinter].mom.R()-trkmcvd.mom.R())",mcmid+gfit);
  ta->Project("t0res","trksegpars_lh[][trkmcvd.iinter].t0-fmod(trkmcvd.time,1695)",mcmid+gfit);
  ta->Project("t0pull","(1.0/trksegpars_lh[][trkmcvd.iinter].t0err)*(trksegpars_lh[][trkmcvd.iinter].t0-fmod(trkmcvd.time,1695))",mcmid+gfit);
  ta->Project("dres","trkhits[].rdrift-trkhitsmc[].dist",ghit+gfit);
  ta->Project("dresg","trkhits[].rdrift-trkhitsmc[].dist",ghit+gdrift+gfit);
  ta->Project("dresb","trkhits[].rdrift-trkhitsmc[].dist",ghit+bdrift+gfit);
  ta->Project("dresp","(trkhits[].rdrift-trkhitsmc[].dist)/trkhits[].uderr",ghit+gfit);
  ta->Project("drespg","(trkhits[].rdrift-trkhitsmc[].dist)/trkhits[].uderr",ghit+gdrift+gfit);
  ta->Project("drespb","(trkhits[].rdrift-trkhitsmc[].dist)/trkhits[].uderr",ghit+bdrift+gfit);

  ta->Project("udoca","abs(trkhits[].udoca)-trkhitsmc[].dist",ghit);
  ta->Project("udocap","(abs(trkhits[].udoca)-trkhitsmc[].dist)/sqrt(trkhits[].udocavar)",gfit+ghit);

  ta->Project("bbqual","trkhits[].bkgqual",gfit+!thit);
  ta->Project("cbqual","trkhits[].bkgqual",gfit+thit);

  psqual->Sumw2();
  gsqual->Sumw2();
  bsqual->Sumw2();
  ta->Project("gsqual","trkhits[].signqual",gfit+ghit+gambig);
  ta->Project("bsqual","trkhits[].signqual",gfit+ghit+bambig);

  psqual->Divide(gsqual,bsqual);

  ta->Project("gdqual","trkhits[].driftqual",gfit+ghit+gdrift);
  ta->Project("bdqual","trkhits[].driftqual",gfit+ghit+bdrift);

  ta->Project("calotres","trkcalohit.tresid",gfit+calo);
  ta->Project("calotpull","trkcalohit.tresid/sqrt(trkcalohit.tresidpvar+trkcalohit.tresidpvar)",gfit+calo);

  uamcdoca->Add(tmcdoca,uamcdoca,1.0,-1.0);
  ngmcdoca->SetFillColor(kBlue);
  nbmcdoca->SetFillColor(kCyan);
  gmcdoca->SetFillColor(kGreen);
  dmcdoca->SetFillColor(kYellow);
  bmcdoca->SetFillColor(kRed);
  ibmcdoca->SetFillColor(kOrange);
  icmcdoca->SetFillColor(kGray);
  umcdoca->SetFillColor(kBlack);
  uamcdoca->SetFillColor(kMagenta);
  rmcdoca->SetFillColor(kCyan);
  THStack* mcdocas = new THStack("mcdocas","Hit State vs MC DOCA;MC DOCA (mm)");
  mcdocas->Add(umcdoca);
//  mcdocas->Add(rmcdoca);
  mcdocas->Add(ibmcdoca);
//  mcdocas->Add(uamcdoca);
  mcdocas->Add(icmcdoca);
  mcdocas->Add(bmcdoca);
  mcdocas->Add(ngmcdoca);
  mcdocas->Add(nbmcdoca);
  mcdocas->Add(dmcdoca);
  mcdocas->Add(gmcdoca);

  ngcdrift->SetFillColor(kBlue);
  nbcdrift->SetFillColor(kCyan);
  gcdrift->SetFillColor(kGreen);
  dcdrift->SetFillColor(kYellow);
  bcdrift->SetFillColor(kRed);
  ibcdrift->SetFillColor(kOrange);
  iccdrift->SetFillColor(kGray);
  ucdrift->SetFillColor(kBlack);
  rcdrift->SetFillColor(kCyan);
  THStack* cdrifts = new THStack("cdrifts","Hit State vs CDrift;CDrift (mm)");
  cdrifts->Add(ucdrift);
  cdrifts->Add(ibcdrift);
  cdrifts->Add(iccdrift);
  cdrifts->Add(bcdrift);
  cdrifts->Add(ngcdrift);
  cdrifts->Add(nbcdrift);
  cdrifts->Add(dcdrift);
  cdrifts->Add(gcdrift);

  ngrdrift->SetFillColor(kBlue);
  nbrdrift->SetFillColor(kCyan);
  grdrift->SetFillColor(kGreen);
  drdrift->SetFillColor(kYellow);
  brdrift->SetFillColor(kRed);
  ibrdrift->SetFillColor(kOrange);
  icrdrift->SetFillColor(kGray);
  urdrift->SetFillColor(kBlack);
  rrdrift->SetFillColor(kCyan);
  THStack* rdrifts = new THStack("rdrifts","Hit State vs RDrift;RDrift (mm)");
  rdrifts->Add(urdrift);
  rdrifts->Add(ibrdrift);
  rdrifts->Add(icrdrift);
  rdrifts->Add(brdrift);
  rdrifts->Add(ngrdrift);
  rdrifts->Add(nbrdrift);
  rdrifts->Add(drdrift);
  rdrifts->Add(grdrift);

  ngudoca->SetFillColor(kBlue);
  nbudoca->SetFillColor(kCyan);
  gudoca->SetFillColor(kGreen);
  dudoca->SetFillColor(kYellow);
  budoca->SetFillColor(kRed);
  ibudoca->SetFillColor(kOrange);
  icudoca->SetFillColor(kGray);
  uudoca->SetFillColor(kBlack);
  rudoca->SetFillColor(kCyan);
  THStack* udocas = new THStack("udocas","Hit State vs UDOCA;UDOCA (mm)");
  udocas->Add(uudoca);
  udocas->Add(ibudoca);
  udocas->Add(icudoca);
  udocas->Add(budoca);
  udocas->Add(ngudoca);
  udocas->Add(nbudoca);
  udocas->Add(dudoca);
  udocas->Add(gudoca);

  amcdoca->SetLineColor(kGreen);
  ardrift->SetLineColor(kBlue);
  audoca->SetLineColor(kRed);
  amcdoca->SetStats(0);
  ardrift->SetStats(0);
  audoca->SetStats(0);
  TLegend* dleg = new TLegend(0.3,0.2,0.6,0.6);
  dleg->AddEntry(amcdoca,"MC true","L");
  dleg->AddEntry(ardrift,"RDrift","L");
  dleg->AddEntry(audoca,"UDOCA","L");

  THStack* dress = new THStack("dress","Drift Resolution;R_{drift}-MC DOCA (mm)");
  dresg->SetFillColor(kGreen);
  dresb->SetFillColor(kRed);
  dress->Add(dresb);
  dress->Add(dresg);
  THStack* dresps = new THStack("dresps","Drift Resolution Pull;(R_{drift}-MC DOCA)/#sigma_{rdrift}");
  drespg->SetFillColor(kGreen);
  drespb->SetFillColor(kRed);
  dresps->Add(drespb);
  dresps->Add(drespg);

  TLegend* rleg = new TLegend(0.1,0.5,0.4,0.9);
  rleg->AddEntry(dresg,"Good drift","F");
  rleg->AddEntry(dresb,"Bad drift","F");

  TLegend* hsleg = new TLegend(0.1,0.1,0.9,0.9);
  char title[80];
  snprintf(title,80,"Correct LR, good drift %.1f %%",gmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(gmcdoca,title,"F");
  snprintf(title,80,"Correct LR, bad drift %.1f %%",dmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(dmcdoca,title,"F");
  snprintf(title,80,"Null LR, bad drift %.1f %%",nbmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(nbmcdoca,title,"F");
  snprintf(title,80,"Null LR, good drift %.1f %%",ngmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(ngmcdoca,title,"F");
  snprintf(title,80,"Wrong LR %.1f %%",bmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(bmcdoca,title,"F");
  snprintf(title,80,"Inactive Ce %.2f %%",icmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(icmcdoca,title,"F");
  snprintf(title,80,"Unassociated Ce %.2f %%",uamcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(uamcdoca,title,"F");
  snprintf(title,80,"Inactive Background %.2f %%",ibmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(ibmcdoca,title,"F");
  //  snprintf(title,80,"Ce Relative %.2f %%",rmcdoca->GetEntries()*100/tmcdoca->GetEntries());
  //  hsleg->AddEntry(rmcdoca,title,"F");
  snprintf(title,80,"Active Background %.2f %%",umcdoca->GetEntries()*100/tmcdoca->GetEntries());
  hsleg->AddEntry(umcdoca,title,"F");

  bbqual->SetLineColor(kRed);
  cbqual->SetLineColor(kGreen);
  bbqual->SetStats(0);
  cbqual->SetStats(0);

  gsqual->SetLineColor(kGreen);
  bsqual->SetLineColor(kRed);

//  psqual->SetStats(0);
  gsqual->SetStats(0);
  bsqual->SetStats(0);

  gdqual->SetLineColor(kGreen);
  bdqual->SetLineColor(kRed);

  gdqual->SetStats(0);
  bdqual->SetStats(0);

  TLegend* bqleg = new TLegend(0.2,0.6,0.7,0.9);
  bqleg->AddEntry(cbqual,"True Primary hit","L");
  bqleg->AddEntry(bbqual,"Background hit","L");

  TLegend* sqleg = new TLegend(0.2,0.7,0.7,0.9);
  sqleg->AddEntry(gsqual,"Correct LR Sign","L");
  sqleg->AddEntry(bsqual,"Incorrect LR Sign","L");

  TLegend* dqleg = new TLegend(0.3,0.2,0.8,0.4);
  dqleg->AddEntry(gdqual,"Good Clustering","L");
  dqleg->AddEntry(bdqual,"Bad Clustering","L");

  TCanvas* hscan = new TCanvas("hscan","hscan",1600,1000);
  hscan->Divide(3,2);
  hscan->cd(1);
  mcdocas->Draw();
  hscan->cd(2);
  cdrifts->Draw();
  hscan->cd(3);
  rdrifts->Draw();
  hscan->cd(4);
  udocas->Draw();
  hscan->cd(5);
  ardrift->Draw();
  amcdoca->Draw("same");
  audoca->Draw("same");
  dleg->Draw();
  hscan->cd(6);
  hsleg->Draw();

  TF1* tf = new TF1("tf","[0]*tan([1]*x)",0.0,1.0);
  tf->SetParameters(psqual->GetMaximum()/10.0,1.571);

  TCanvas* qualcan = new TCanvas("qualcan","qualcan",1500,1000);
  qualcan->Divide(3,2);
  qualcan->cd(1);
  fcon->Draw();
  qualcan->cd(2);
  gPad->SetLogy();
  cbqual->SetMinimum(0.5);
  cbqual->Draw();
  bbqual->Draw("same");
  bqleg->Draw();
  qualcan->cd(3);
  gPad->SetLogy();
  gsqual->SetMinimum(0.5);
  gsqual->SetMaximum(max(gsqual->GetMaximum(),bsqual->GetMaximum()));
  gsqual->Draw();
  bsqual->Draw("same");
  sqleg->Draw();
  qualcan->cd(4);
  gPad->SetLogy();
  gdqual->SetMinimum(0.5);
  bdqual->SetMaximum(max(gdqual->GetMaximum(),bdqual->GetMaximum()));
  gdqual->Draw();
  bdqual->Draw("same");
  dqleg->Draw();
  qualcan->cd(5);
//  psqual->Fit(tf,"","",0.02,0.98);
//  auto pad = qualcan->GetPad(6);
//  pad->Divide(2,1);
//  pad->cd(1);
  calotres->Fit("gaus");
  qualcan->cd(6);
//  pad->cd(2);
  calotpull->Fit("gaus");

  TCanvas* fitcan = new TCanvas("fitcan","fitcan",1600,1000);
  fitcan->Divide(3,2);
  fitcan->cd(1);
  entmomres->Fit("gaus");
  fitcan->cd(2);
  midmomres->Fit("gaus");
  fitcan->cd(3);
  xitmomres->Fit("gaus");
  fitcan->cd(4);
  midmompull->Fit("gaus");
  fitcan->cd(5);
  t0res->Fit("gaus");
  fitcan->cd(6);
  t0pull->Fit("gaus");

  TCanvas* rescan = new TCanvas("rescan","rescan",1200,1000);
  rescan->Divide(2,2);
  rescan->cd(1);
  dress->Draw();
  dres->SetStats(true);
  //dresg->SetFillColor(0);
  dres->Fit("gaus","q","sames");
  rleg->Draw();
  rescan->cd(2);
  dresps->Draw();
  dresp->SetStats(true);
  //drespg->SetFillColor(0);
  dresp->Fit("gaus","q","sames");
  rescan->cd(3);
  udoca->Fit("gaus");
  rescan->cd(4);
  udocap->Fit("gaus");

  string ssuf(savesuffix);
  string term(".png");
  if(!ssuf.empty()){
    string hsfile = string("hscan_") + ssuf + term;
    hscan->Draw();
    hscan->SaveAs(hsfile.c_str());
    string fitfile = string("fitcan_") + ssuf + term;
    fitcan->Draw();
    fitcan->SaveAs(fitfile.c_str());
    string resfile = string("rescan_") + ssuf + term;
    rescan->Draw();
    rescan->SaveAs(resfile.c_str());
    string qualfile = string("qualcan_") + ssuf + term;
    qualcan->Draw();
    qualcan->SaveAs(qualfile.c_str());
  }
}
void DriftDiagFile(const char* file,const char* ssuf="") {
  TFile* tf = new TFile(file);
  TTree* ta = (TTree*)tf->Get("EventNtuple/ntuple");
  DriftDiag(ta,ssuf);
}

void DriftDiagChain(const char* files,const char* cname="EventNtuple/ntuple",const char* ssuf="") {
  TChain* ta = new TChain(cname);
  FillChain(ta,files);
  DriftDiag(ta,ssuf);
}

