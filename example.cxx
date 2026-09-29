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
#include <TLegend.h>

// ============================================================================
// Working points, mirrored 1:1 from TestingMacro_Combined_Podio.cxx so that
// the efficiency numbers/plots produced here match that macro exactly.
// P_SPLIT_GEV itself comes straight from MuonID.cxx (same translation unit,
// since it's #include-d as source above).
// ============================================================================
static const float MUON_ID_CUT_HIGHP = 0.5f; // from TestingMacro_Podio.cxx
static const float MUON_ID_CUT_LOWP  = 0.5f; // placeholder, from TestingMacro_Fixed_Podio.cxx

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

    //const std::string pattern ="/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";

    const std::string pattern ="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";

    std::vector<std::string> infiles = ExpandGlob(pattern);

    if (infiles.empty()) {
        std::cerr << "No files found matching pattern: " << pattern << std::endl;
        return;
    }
    std::cout << "Found " << infiles.size() << " files to process." << std::endl;

    podio::ROOTReader reader;
    reader.openFiles(infiles);

    // -------- histograms --------
    // FIX #8: use the SAME eta range for MC / reco / found / id everywhere,
    // matching the acceptance cut applied below (-1.25 < eta < 3.5, crack
    // removed), so all numerators and denominators cover the same phase
    // space.
    //
    // Three stages, matching TestingMacro_Combined_Podio.cxx exactly:
    //   mc    : true muons at generator level, in acceptance                (denominator #1)
    //   found : reconstructed track in acceptance AND MuonID() reports
    //           hasDetectorInfo=true -- i.e. it left enough of a signal in
    //           calo/ToF to actually be scored. This is exactly the "Total"
    //           condition in TestingMacro_Combined_Podio.cxx (keep==true).
    //   id    : found AND P(muon) > cut (regime-dependent cut, same as
    //           TestingMacro_Combined_Podio.cxx). This is exactly the
    //           "Passed" condition there.
    //
    // reco (all reconstructed tracks in acceptance, regardless of whether
    // MuonID could actually score them) is kept too, only as an extra
    // sanity-check number -- it is NOT used for the two efficiencies we
    // plot, since TestingMacro_Combined_Podio.cxx doesn't use it either.
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

    TH1F *h_eta_mc    = new TH1F("h_eta_mc",    "; #eta; events", 25, -1.25, 3.5);
    TH1F *h_eta_reco  = new TH1F("h_eta_reco",  "; #eta; events", 25, -1.25, 3.5);
    TH1F *h_eta_found = new TH1F("h_eta_found", "; #eta; events", 25, -1.25, 3.5);
    TH1F *h_eta_id    = new TH1F("h_eta_id",    "; #eta; events", 25, -1.25, 3.5);

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

    // -------- event loop --------
    const auto n_events = 1000;
    for (size_t iev = 0; iev < n_events; ++iev) {
        const auto event = podio::Frame(reader.readNextEntry("events"));
        if (iev % 1000 == 0)
            std::cout << "Processing event " << iev << " / " << n_events << std::endl;

        // MC Particles: denominator for the "found" (reco + capable of ID)
        // efficiency. Same acceptance cut as applied to reco below, so eta
        // ranges match (FIX #8).
        const auto &mc_parts = event.get<edm4hep::MCParticleCollection>("MCParticles");
        for (const auto &mcp : mc_parts) {
            const auto mom = mcp.getMomentum();
            const int pdg = mcp.getPDG();
            const double mc_eta = edm4hep::utils::eta(mom);

            bool mc_in_acceptance = (mc_eta > -1.0 && mc_eta < 3.5);
            bool mc_in_crack = (mc_eta > 1.0 && mc_eta < 1.3);

            if (std::abs(pdg) == 13 && mc_in_acceptance && !mc_in_crack) {
                h_mom_mc->Fill(edm4hep::utils::magnitude(mom));
                h_eta_mc->Fill(mc_eta);
            }
        }

        // FIX #6: use ReconstructedChargedParticles, not ReconstructedParticles.
        // MuonID() needs a charged track (it scores tracks, not neutral
        // clusters), so iterating over the generic ReconstructedParticles
        // collection wastes denominator entries on particles MuonID() will
        // trivially reject regardless of PID quality.
        const auto &reco_parts = event.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");

        // FIX #7: truth-match each reconstructed charged particle to its MC
        // particle, exactly like TestingMacro_Combined_Podio.cxx does via
        // trackAssocs[particle].getSim().
        //
        // NOTE: for now we only run this on the pure-muon sample (see
        // `pattern` above), so every reconstructed charged particle in
        // acceptance is treated as a true muon -- exactly the same
        // assumption TestingMacro_Combined_Podio.cxx makes for its "Muons"
        // file (it never checks simPart's PDG either). If/when a mixed or
        // pion-only sample is run through this macro, uncomment the PDG
        // check below to restrict "found"/"id" filling to true muons only.
        const auto &trackAssocs = event.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedChargedParticleAssociations");

        for (size_t i = 0; i < reco_parts.size(); ++i) {
            const auto rcp = reco_parts[i];

            if (i >= trackAssocs.size()) continue; // no truth link available
            const auto simPart = trackAssocs[i].getSim();
            //if (std::abs(simPart.getPDG()) != 13) continue; // not a true muon -> skip (pure-muon sample only for now)

            const auto mom = rcp.getMomentum();
            const double p   = edm4hep::utils::magnitude(mom);
            const double eta = edm4hep::utils::eta(mom);

            bool in_acceptance = (eta > -1.0 && eta < 3.5);
            bool in_crack = (eta > 1.0 && eta < 1.3);

            if (!in_acceptance || in_crack) continue;

            // "reco": every reconstructed charged track in acceptance,
            // regardless of whether it can actually be scored. Sanity-check
            // number only, not used in the two plotted efficiencies.
            h_mom_reco->Fill(p);
            h_eta_reco->Fill(eta);

            // "found": reco AND MuonID() had enough detector info to score
            // it. This is exactly the "keep" / "Total" condition in
            // TestingMacro_Combined_Podio.cxx.
            bool hasDetectorInfo = false;
            const float Pmu = MuonID(rcp, &event, &hasDetectorInfo);

            if (!hasDetectorInfo) continue; // not found -> not scoreable, skip (matches "keep=false" there)

            h_mom_found->Fill(p);
            h_eta_found->Fill(eta);

            // Regime-dependent working point, same selection logic as
            // TestingMacro_Combined_Podio.cxx (float cut = (trackP <
            // P_SPLIT_GEV) ? MUON_ID_CUT_LOWP : MUON_ID_CUT_HIGHP;).
            const float cut = (p < P_SPLIT_GEV) ? MUON_ID_CUT_LOWP : MUON_ID_CUT_HIGHP;

            // "id": found AND P(muon) > cut. Exactly the "Passed" condition
            // in TestingMacro_Combined_Podio.cxx.
            if (Pmu > cut) {
                h_mom_id->Fill(p);
                h_eta_id->Fill(eta);
            }
        }
    }

    // -------- efficiencies --------
    // 1) "Found" efficiency: MC truth muon -> reconstructed AND capable of
    //    being identified (left a usable signal in calo/ToF).
    // 2) "Muon ID" efficiency: found -> correctly identified as a muon.
    //    This is the SAME ratio (Passed/Total) as
    //    h_Muon_Efficiency_vs_P / h_Muon_Efficiency_vs_Eta in
    //    TestingMacro_Combined_Podio.cxx.
    double eff_found = h_mom_found->Integral() / h_mom_mc->Integral();
    double eff_id     = h_mom_id->Integral()    / h_mom_found->Integral();

    cout << "Found efficiency  (found/mc, true muons only)    : " << eff_found << "\n";
    cout << "Muon ID efficiency (id/found, true muons only)   : " << eff_id    << "\n";
    cout << "reco/mc (no detector-info filter) : "
         << h_mom_reco->Integral() / h_mom_mc->Integral() << "\n";

    // -------- draw histograms --------
    gStyle->SetOptStat(0);

    TLegend *legend = new TLegend(0.55, 0.55, 0.85, 0.75);
    legend->AddEntry(h_mom_mc,    "MC truth", "l");
    legend->AddEntry(h_mom_found, "Found (reco + scoreable)", "l");
    legend->AddEntry(h_mom_id,    "Identified as muon", "l");
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextSize(0.03);

    TCanvas *c1 = new TCanvas("c1", "c1", 800, 600);
    h_mom_mc->SetLineColor(kRed);
    h_mom_mc->Draw("HIST");
    h_mom_found->SetLineColor(kGreen);
    h_mom_found->Draw("HIST SAME");
    h_mom_id->SetLineColor(kBlue);
    h_mom_id->Draw("HIST SAME");
    legend->Draw();

    c1->SaveAs("muID.pdf");

    TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);
    c2->SaveAs("muID_efficiency.pdf[");

    // Page 1: found/mc vs p
    h_eff_found_vs_mc_mom->Divide(h_mom_found, h_mom_mc, 1.0, 1.0, "B");
    h_eff_found_vs_mc_mom->SetTitle("Found Efficiency (reco + scoreable) / MC;Mom [GeV/c];Efficiency");
    h_eff_found_vs_mc_mom->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->Clear();

    // Page 2: id/found vs p -- 1:1 with h_Muon_Efficiency_vs_P in
    // TestingMacro_Combined_Podio.cxx
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

    // Page 4: id/found vs eta -- 1:1 with h_Muon_Efficiency_vs_Eta in
    // TestingMacro_Combined_Podio.cxx
    h_eff_id_vs_found_eta->Divide(h_eta_id, h_eta_found, 1.0, 1.0, "B");
    h_eff_id_vs_found_eta->SetTitle("Muon ID Efficiency (id/found);#eta;Efficiency");
    h_eff_id_vs_found_eta->Draw("e1");
    c2->SaveAs("muID_efficiency.pdf");
    c2->SaveAs("muID_efficiency.pdf]");

    // -------- Save all histograms to a ROOT file --------
    TFile *outFile = new TFile("MuonID_Histograms.root", "RECREATE");
    outFile->cd(); // Switch to the output file directory

    // Count histograms
    h_mom_mc->Write();
    h_mom_reco->Write();
    h_mom_found->Write();
    h_mom_id->Write();

    h_eta_mc->Write();
    h_eta_reco->Write();
    h_eta_found->Write();
    h_eta_id->Write();

    // Efficiency histograms
    h_eff_found_vs_mc_mom->Write();
    h_eff_id_vs_found_mom->Write();

    h_eff_found_vs_mc_eta->Write();
    h_eff_id_vs_found_eta->Write();

    outFile->Close();
    delete outFile;

    std::cout << "Histograms successfully saved to MuonID_Histograms.root" << std::endl;

}