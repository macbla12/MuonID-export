#include <glob.h>
#include "podio/ROOTReader.h"
#include "podio/Frame.h"
#include "edm4hep/utils/vector_utils.h"
#include "edm4hep/utils/kinematics.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/ReconstructedParticleCollection.h"
#include "edm4eic/ClusterCollection.h"
#include "edm4hep/Vector3f.h"
#include "edm4hep/Vector2f.h"
#include "edm4eic/MCRecoParticleAssociationCollection.h"
#include "MuonID.cxx"
#include <TStyle.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TGaxis.h>
#include <TPad.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>

// ============================================================================
// Working points. P_SPLIT_GEV pochodzi z MuonID.cxx (ten sam translation unit).
// ============================================================================
static const float MUON_ID_CUT_HIGHP = 0.5f;
static const float MUON_ID_CUT_LOWP  = 0.5f;

std::vector<std::string> ExpandGlob(const std::string &pattern)
{
    std::vector<std::string> filenames;
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    globfree(&glob_result);
    return filenames;
}

void example() {

    //const std::string pattern ="/run/media/epic/Data/Background/Muon/Podio-1.7.99/*.root";

    const std::string pattern ="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_07*.root";

    double MOMNOW=0.0;

    std::vector<std::string> infiles = ExpandGlob(pattern);

    if (infiles.empty()) {
        std::cerr << "No files found matching pattern: " << pattern << std::endl;
        return;
    }
    std::cout << "Found " << infiles.size() << " files to process." << std::endl;

    podio::ROOTReader reader;
    reader.openFiles(infiles);

    // -------- histograms --------
    TH1F *h_mom_mc    = new TH1F("h_mom_mc",    "; p [GeV]; events", 25, 0, 4);
    TH1F *h_mom_reco  = new TH1F("h_mom_reco",  "; p [GeV]; events", 25, 0, 4);
    TH1F *h_mom_found = new TH1F("h_mom_found", "; p [GeV]; events", 25, 0, 4);
    TH1F *h_mom_id    = new TH1F("h_mom_id",    "; p [GeV]; events", 25, 0, 4);

    TH1D *h_eff_found_vs_mc_mom = (TH1D*)h_mom_found->Clone("h_eff_found_vs_mc_mom");
    h_eff_found_vs_mc_mom->Reset();
    h_eff_found_vs_mc_mom->Sumw2();

    TH1D *h_eff_id_vs_found_mom = (TH1D*)h_mom_id->Clone("h_eff_id_vs_found_mom");
    h_eff_id_vs_found_mom->Reset();
    h_eff_id_vs_found_mom->Sumw2();

    h_mom_mc->Sumw2();
    h_mom_reco->Sumw2();
    h_mom_found->Sumw2();
    h_mom_id->Sumw2();

    TH1F *h_eta_mc    = new TH1F("h_eta_mc",    "; #eta; events", 25, -3.5, 3.5);
    TH1F *h_eta_reco  = new TH1F("h_eta_reco",  "; #eta; events", 25, -3.5, 3.5);
    TH1F *h_eta_found = new TH1F("h_eta_found", "; #eta; events", 25, -3.5, 3.5);
    TH1F *h_eta_id    = new TH1F("h_eta_id",    "; #eta; events", 25, -3.5, 3.5);

    TH1D *h_eff_found_vs_mc_eta = (TH1D*)h_eta_found->Clone("h_eff_found_vs_mc_eta");
    h_eff_found_vs_mc_eta->Reset();
    h_eff_found_vs_mc_eta->Sumw2();

    TH1D *h_eff_id_vs_found_eta = (TH1D*)h_eta_id->Clone("h_eff_id_vs_found_eta");
    h_eff_id_vs_found_eta->Reset();
    h_eff_id_vs_found_eta->Sumw2();

    h_eta_mc->Sumw2();
    h_eta_reco->Sumw2();
    h_eta_found->Sumw2();
    h_eta_id->Sumw2();

    // -------- response histograms --------
    TH1F *h_resp_all  = new TH1F("h_resp_all",  "P(muon) response (all scored);P(muon);Counts", 100, 0, 1);
    TH1F *h_resp_low  = new TH1F("h_resp_low",  "P(muon) response, Low model (p<=P_SPLIT);P(muon);Counts", 100, 0, 1);
    TH1F *h_resp_high = new TH1F("h_resp_high", "P(muon) response, High model (p>P_SPLIT);P(muon);Counts", 100, 0, 1);
    TH2F *h_resp_vs_p = new TH2F("h_resp_vs_p", "P(muon) vs p;p [GeV/c];P(muon)", 40, 0, 4, 100, 0, 1);

    // liczniki diagnostyczne
    long n_charged = 0, n_scored = 0, n_noinfo = 0, n_pass = 0;

    vector<string> collections = { 
        "ReconstructedParticles", "MCParticles", "ReconstructedParticleAssociations",
        "HcalBarrelTrackClusterMatches", "HcalEndcapNTrackClusterMatches", "LFHCALTrackClusterMatches",
        "EcalBarrelClusters", "EcalBarrelImagingRecHits", "EcalBarrelScFiRecHits",
        "EcalEndcapNClusters", "EcalEndcapNRecHits", "EcalEndcapPClusters", "EcalEndcapPRecHits",
        "HcalBarrelClusters", "HcalBarrelRecHits", "HcalEndcapNClusters", "HcalEndcapNMergedHits",
        "LFHCALClusters", "LFHCALRecHits",
        "TOFBarrelSharedRecHits", "TOFEndcapSharedRecHits", "CentralTrackSegments",
        "CentralCKFTracks","CentralTrackSeeds",
    };

    // -------- event loop --------
    const auto n_events = 30000;
    for (size_t iev = 0; iev < n_events; ++iev) {

        const auto event = podio::Frame(reader.readEntry("events", iev, collections));

        // MC particles: mianownik dla "found"
        const auto &mc_parts = event.get<edm4hep::MCParticleCollection>("MCParticles");
        for (const auto &mcp : mc_parts) {

            const auto mom = mcp.getMomentum();
            if (edm4hep::utils::magnitude(mom) < MOMNOW) continue;
            const int pdg = mcp.getPDG();
            const double mc_eta = edm4hep::utils::eta(mom);

            if (std::abs(pdg) == 13) { // true muons only
                h_mom_mc->Fill(edm4hep::utils::magnitude(mom));
                h_eta_mc->Fill(mc_eta);
            }
        }

        const auto &reco_parts = event.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

        for (size_t i = 0; i < reco_parts.size(); ++i) {
            const auto rcp = reco_parts[i];
            if (rcp.getCharge() == 0) continue; // skip neutrals
            const auto mom = rcp.getMomentum();
            const double p   = edm4hep::utils::magnitude(mom);
            if (p<MOMNOW) continue;
            const double eta = edm4hep::utils::eta(mom);

            // "reco": kazdy naladowany slad, sanity check
            h_mom_reco->Fill(p);
            h_eta_reco->Fill(eta);

            // "found": reco AND MuonID mialo dosc info, zeby ocenic
            bool hasDetectorInfo = false;
            const float Pmu = MuonID(rcp, &event, &hasDetectorInfo);
            ++n_charged;

            if (!hasDetectorInfo) { ++n_noinfo; continue; }
            ++n_scored;

            // response
            h_resp_all->Fill(Pmu);
            h_resp_vs_p->Fill(p, Pmu);
            if (p <= P_SPLIT_GEV) h_resp_low->Fill(Pmu);
            else                  h_resp_high->Fill(Pmu);

            h_mom_found->Fill(p);
            h_eta_found->Fill(eta);

            const float cut = (p < P_SPLIT_GEV) ? MUON_ID_CUT_LOWP : MUON_ID_CUT_HIGHP;

            // "id": found AND P(muon) > cut
            if (Pmu > cut) {
                ++n_pass;
                h_mom_id->Fill(p);
                h_eta_id->Fill(eta);
            }
        }
    }

    // -------- efficiencies --------
    double eff_found = h_mom_found->Integral() / h_mom_mc->Integral();
    double eff_id    = h_mom_id->Integral()    / h_mom_found->Integral();

    cout << "Found efficiency  (found/mc, true muons only)    : " << eff_found << "\n";
    cout << "Muon ID efficiency (id/found, true muons only)   : " << eff_id    << "\n";
    cout << "reco/mc (no detector-info filter) : "
         << h_mom_reco->Integral() / h_mom_mc->Integral() << "\n";

    cout << "charged tracks: " << n_charged
         << " | no detector info: " << n_noinfo
         << " | scored: " << n_scored
         << " | passed cut: " << n_pass << "\n";
    cout << "Response mean: " << h_resp_all->GetMean()
         << " (low: " << h_resp_low->GetMean() << ", high: " << h_resp_high->GetMean() << ")\n";

    // -------- draw histograms --------
    gStyle->SetOptStat(0);

    // Response
    TCanvas *c3 = new TCanvas("c3", "c3", 1200, 500);
    c3->Divide(3, 1);
    c3->cd(1); gPad->SetLogy(); h_resp_all->Draw("HIST");
    c3->cd(2); gPad->SetLogy();
    h_resp_low->SetLineColor(kRed);   h_resp_low->Draw("HIST");
    h_resp_high->SetLineColor(kBlue); h_resp_high->Draw("HIST SAME");
    c3->cd(3); h_resp_vs_p->Draw("COLZ");
    c3->SaveAs("muID_response.pdf");

    // Efficiency
    TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);
    c2->SaveAs("muID_efficiency.pdf[");

    // Page 1: found/mc vs p
    h_eff_found_vs_mc_mom->Divide(h_mom_found, h_mom_mc, 1.0, 1.0, "B");
    h_eff_found_vs_mc_mom->SetTitle("Found Efficiency (reco + scoreable) / MC;Mom [GeV/c];Efficiency");
    h_eff_found_vs_mc_mom->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->Clear();

    // Page 2: id/found vs p
    h_eff_id_vs_found_mom->Divide(h_mom_id, h_mom_found, 1.0, 1.0, "B");
    h_eff_id_vs_found_mom->SetTitle("Muon ID Efficiency (id/found);Mom [GeV/c];Efficiency");
    h_eff_id_vs_found_mom->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->Clear();

    // Page 3: found/mc vs eta
    h_eff_found_vs_mc_eta->Divide(h_eta_found, h_eta_mc, 1.0, 1.0, "B");
    h_eff_found_vs_mc_eta->SetTitle("Found Efficiency (reco + scoreable) / MC;#eta;Efficiency");
    h_eff_found_vs_mc_eta->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->Clear();

    // Page 4: id/found vs eta
    h_eff_id_vs_found_eta->Divide(h_eta_id, h_eta_found, 1.0, 1.0, "B");
    h_eff_id_vs_found_eta->SetTitle("Muon ID Efficiency (id/found);#eta;Efficiency");
    h_eff_id_vs_found_eta->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->SaveAs("muID_efficiency.pdf]");

    // -------- Save all histograms to a ROOT file --------
    TFile *outFile = new TFile("MuonID_Histograms.root", "RECREATE");
    outFile->cd();

    h_mom_mc->Write();
    h_mom_reco->Write();
    h_mom_found->Write();
    h_mom_id->Write();

    h_eta_mc->Write();
    h_eta_reco->Write();
    h_eta_found->Write();
    h_eta_id->Write();

    h_eff_found_vs_mc_mom->Write();
    h_eff_id_vs_found_mom->Write();

    h_eff_found_vs_mc_eta->Write();
    h_eff_id_vs_found_eta->Write();

    h_resp_all->Write();
    h_resp_low->Write();
    h_resp_high->Write();
    h_resp_vs_p->Write();

    outFile->Close();
    delete outFile;

    std::cout << "Histograms successfully saved to MuonID_Histograms.root" << std::endl;

}