#include <glob.h>
#include "podio/ROOTReader.h"
#include "podio/Frame.h"
#include "edm4hep/utils/vector_utils.h"
#include "edm4hep/utils/kinematics.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/ReconstructedParticleCollection.h"
#include "edm4eic/ClusterCollection.h"
#include "edm4hep/Vector3f.h"
#include "edm4hep/Vector2f.h" // warto dodać profilaktycznie
#include "edm4eic/MCRecoParticleAssociationCollection.h"
#include "MuonID.cxx"
#include <TStyle.h>
#include <TCanvas.h>
#include <TGaxis.h>
#include <TPad.h>
#include <TH1.h>
#include <TLegend.h>

// podio::ROOTReader::openFiles() NIE obsluguje globow ("*") tak jak
// TChain::Add() - trzeba je rozwinac recznie zanim przekazemy liste do readera.
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

    // Prog decyzyjny na P(muon) - DOPASUJ do tego, na czym ustalales
    // punkt pracy (working point) dla dwoch modeli ONNX w MuonID.cxx.
    const double MUON_ID_CUT = 0.5;

    const std::string pattern = "/run/media/epic/Data/Background/Muons/Continuous/reco_*.root";
    //const std::string pattern ="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_221.root";
    //const std::string pattern ="/run/media/epic/Data/Tau/reco/Energy_10x275/old/double_pi/recoDoublePi.root";

   //const std::string pattern ="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Electrons.root";
   //const std::string pattern ="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Kaons.root";
   //const std::string pattern ="/run/media/epic/Data/Background/SingleParticles/SingleFiles/Protons.root";
    std::vector<std::string> infiles = ExpandGlob(pattern);

    if (infiles.empty()) {
        std::cerr << "Nie znaleziono zadnych plikow pasujacych do wzorca: " << pattern << std::endl;
        return;
    }
    std::cout << "Znaleziono " << infiles.size() << " plikow do przetworzenia." << std::endl;

    podio::ROOTReader reader;
    reader.openFiles(infiles);

    // -------- histograms --------
    TH1F *h_mom_mc = new TH1F("h_mom_mc", "; p [GeV]; events", 100, 0, 20);
    TH1F *h_mom_reco = new TH1F("h_mom_reco", "; p [GeV]; events", 100, 0, 20);
    TH1D *h_eff = (TH1D*)h_mom_reco->Clone("h_eff");
    h_eff->Reset(); 
    h_eff->Sumw2();
    h_mom_reco->Sumw2();
    h_mom_mc->Sumw2();

    // -------- event loop --------
    const auto n_events = 100000;
    int qt = 0;
    for (size_t iev = 0; iev < n_events; ++iev) {
        const auto event = podio::Frame(reader.readNextEntry("events"));
        if (iev % 1000 == 0)
            std::cout << "Processing event " << iev << " / " << n_events << std::endl;

        // MC Particles
        const auto &mc_parts = event.get<edm4hep::MCParticleCollection>("MCParticles");
        for (const auto &mcp : mc_parts) {
            const auto mom = mcp.getMomentum();
            const int pdg = mcp.getPDG();
            if (std::abs(edm4hep::utils::eta(mom)) < 3.5 && std::sqrt(mom.x * mom.x + mom.y * mom.y) > 0.3) {
                if (std::abs(pdg) == 13) { // mion
                    h_mom_mc->Fill(edm4hep::utils::magnitude(mom));
                }
            }
        }

        // Reconstructed Particles
        const auto &reco_parts = event.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");
        for (const auto &rcp : reco_parts) {

            const auto mom = rcp.getMomentum();

            if (std::abs(edm4hep::utils::eta(mom)) < 3.5 && std::sqrt(mom.x * mom.x + mom.y * mom.y) > 0.3) {
                if (rcp.getTracks().size() == 0 || rcp.getClusters().size() == 0)
                    qt += 1;
                double p_value = MuonID(rcp, &event);
                if (p_value > MUON_ID_CUT)
                    h_mom_reco->Fill(edm4hep::utils::magnitude(mom));
            }
        }
    }

    // -------- efficiency --------
    double eff = h_mom_reco->Integral() / h_mom_mc->Integral();
    cout << "No track or cluster : " << (double)qt / h_mom_mc->Integral() << " %\n";
    cout << "Efficiency (tot) : " << eff << " %\n";

    // -------- draw histograms --------
    gStyle->SetOptStat(0);

    TLegend *legend = new TLegend(0.6, 0.6, 0.8, 0.7);
    legend->AddEntry(h_mom_mc, "MC truth", "l");
    legend->AddEntry(h_mom_reco, "Reco", "l");
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextSize(0.03);
    legend->Draw();

    TCanvas *c1 = new TCanvas("c1", "c1", 800, 600);
    h_mom_mc->SetLineColor(kRed);
    h_mom_mc->Draw("HIST");
    h_mom_reco->SetLineColor(kBlue);
    h_mom_reco->Draw("HIST SAME");

    
    c1->SaveAs("muID.pdf");

    TCanvas *c2 = new TCanvas("c2", "c2", 800, 600);

    h_eff->Divide(h_mom_reco, h_mom_mc, 1.0, 1.0, "B");

    h_eff->SetTitle("Muon ID Efficiency;p [GeV/c];Efficiency");
    h_eff->Draw("e1");

    c2->SaveAs("muID_efficiency.pdf");
}