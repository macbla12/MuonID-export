// MuonID.cxx
//
// UWAGA - ZALOZENIA DO WERYFIKACJI (przed uzyciem na produkcji):
//
//   1) P_THRESHOLD_GEV        - prog pedu rozdzielajacy model low-P / high-P.
//   2) Sciezki .onnx           - CalorimetryHits/ONNX/xgb_muonID.onnx (high-P, 30 cech,
//                                 calo-only, kolejnosc jak w Twoim TestingMacro.cxx /
//                                 build_raw_features) oraz ToF/ONNX/xgb_muonID.onnx
//                                 (low-P, 21 cech, calo(uproszczone)+ToF+kinematyka,
//                                 kolejnosc jak w TestingMacro_Fixed.cxx /
//                                 build_raw_features_from_components).
//   3) Nazwy kolekcji ToF      - zakladam "TOFBarrelRecHits" / "TOFEndcapRecHits" typu
//                                 edm4eic::TrackerHitCollection (getPosition(), getTime()).
//                                 PODMIEN na wlasciwe nazwy/typ jesli inne w Twoim EICrecon.
//   4) ToFSim / mapa pola      - dolaczona z ToFSim.cxx (RK4, propagacja w mm). Sciezka
//                                 do pliku z mapa pola jest zaszyta w ToFSim.cxx - upewnij
//                                 sie, ze jest poprawna na maszynie, na ktorej to dziala.
//   5) dR_cut / dist_cut ToF   - wartosci 0.8 / 6.0 mm przepisane z TestingMacro_Fixed.cxx,
//                                 do przetestowania czy sa optymalne w tym kontekscie.
//   6) MISSING_SENTINEL = -999 - dla brakujacych cech ECal/HCal/ToF, zgodnie z Python RAW_COLS.
//   7) Hipoteza masy mion PDG  - do budowy TLorentzVector.
//
// Jesli ktorekolwiek z powyzszych zalozen jest nieprawidlowe, model low-P dostanie
// zle uporzadkowany / zle wypelniony wektor wejsciowy i da bezsensowne predykcje
// (lub ONNX Runtime rzuci wyjatkiem o niezgodnym shape).

#include "MuonID.hpp"

#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <onnxruntime_cxx_api.h>

// Wymagane bezposrednio - w odroznieniu od edm4eic::Cluster/Track/TrackClusterMatch
// (ktore sa juz gdzies wciagniete transitywnie), typ edm4eic::TrackerHitCollection
// (uzywany do odczytu hitow ToF) nie byl dotad wlaczony w tym pliku, co powodowalo
// niekompletny typ i blad "failed template argument deduction" w Frame::get<T>().
#include <edm4eic/TrackerHitCollection.h>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

// Dolaczamy implementacje propagacji RK4 / mapy pola (ToFSim, c_light).
// Zaklada, ze ToFSim.cxx lezy w tym samym katalogu (tak jak w TestingMacro_Fixed.cxx).
#include "ToFSim.cxx"

namespace MuonID_Detail {

static std::unique_ptr<Ort::Env>        g_ort_env;
static std::unique_ptr<Ort::Session>    g_session_lowP;
static std::unique_ptr<Ort::Session>    g_session_highP;
static std::unique_ptr<Ort::MemoryInfo> g_mem_info;

// Prog pedu [GeV/c] rozdzielajacy model "niskoenergetyczny" od
// "wysokoenergetycznego". DOPASUJ do tego, na czym trenowales oba ONNX-y.
static constexpr double P_THRESHOLD_GEV = 1.0;

static constexpr float MISSING_SENTINEL = -999.f;

// Ciecia dopasowania hitow ToF do sladu (przepisane z TestingMacro_Fixed.cxx).
static constexpr double TOF_DR_CUT_BARREL   = 0.8;
static constexpr double TOF_DR_CUT_ENDCAP   = 0.8;
static constexpr double TOF_DIST_CUT_BARREL = 6.0;  // mm
static constexpr double TOF_DIST_CUT_ENDCAP = 6.0;  // mm

// =====================================================================
// Cechy kalorymetryczne (hit-level) - bez zmian wzgledem oryginalu.
// =====================================================================
struct CaloFeatures
{
    float Energy = 0.f;
    float Number = 0.f;
    float EoverP = 0.f;
    float AvgHitEnergy = 0.f;
    float SpreadPhi = 0.f;
    float SpreadEta = 0.f;
    float SpreadR = 0.f;
    float MaxHitFrac = 0.f;
    float EnergyStdDev = 0.f;
    float EnergyConcentration = 0.f;
    float R_Disp = 0.f;
    float R_DispWeighted = 0.f;
    float Eta_DispWeighted = 0.f;
    float Phi_DispWeighted = 0.f;
};

template <typename ClusterCollection>
CaloFeatures ComputeCaloFeaturesFromClusters(const ClusterCollection& clusters,
                                              const TLorentzVector& Partic,
                                              double trackP)
{
    CaloFeatures f;

    double sumEnergy = 0.0;
    double maxHitE = -1.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;

    std::vector<double> R, dEta, dPhi, E;

    for (const auto& cluster : clusters)
    {
        sumEnergy += cluster.getEnergy();

        for (const auto& hit : cluster.getHits())
        {
            double HitE = hit.getEnergy();
            auto pos = hit.getPosition();

            TVector3 hitVec(pos.x, pos.y, pos.z);
            double hitEta = hitVec.Eta();
            double hitPhi = hitVec.Phi();
            double Rval = std::sqrt(pos.x * pos.x + pos.y * pos.y);

            E.push_back(HitE);
            R.push_back(Rval);
            dEta.push_back(hitEta - Partic.Eta());
            dPhi.push_back(TVector2::Phi_mpi_pi(hitPhi - Partic.Phi()));

            phiMin = std::min(phiMin, hitPhi); phiMax = std::max(phiMax, hitPhi);
            etaMin = std::min(etaMin, hitEta); etaMax = std::max(etaMax, hitEta);
            Rmin   = std::min(Rmin, Rval);     Rmax   = std::max(Rmax, Rval);

            if (HitE > maxHitE) maxHitE = HitE;
        }
    }

    int n = static_cast<int>(E.size());
    if (sumEnergy <= 0.0 || n == 0) return f; // brak hitow -> cechy zostaja 0

    double meanE = sumEnergy / n;
    double sumSqDevE = 0.0, sumE2 = 0.0;
    for (double e : E) { double d = e - meanE; sumSqDevE += d * d; sumE2 += e * e; }
    double energyStdDev = std::sqrt(sumSqDevE / n);
    double energyConcentration = sumE2 / (sumEnergy * sumEnergy);

    double sumRw = 0.0;
    for (int k = 0; k < n; ++k) sumRw += R[k] * E[k];
    double meanR_w = sumRw / sumEnergy;

    double sumR2diff = 0.0, sumR2diffW = 0.0;
    double sumDEta2diffW = 0.0, sumDPhi2diffW = 0.0;
    for (int k = 0; k < n; ++k)
    {
        double dR = R[k] - meanR_w;
        sumR2diff  += dR * dR;
        sumR2diffW += dR * dR * E[k];
        sumDEta2diffW += dEta[k] * dEta[k] * E[k];
        sumDPhi2diffW += dPhi[k] * dPhi[k] * E[k];
    }

    f.Energy               = static_cast<float>(sumEnergy);
    f.Number               = static_cast<float>(n);
    f.EoverP                = static_cast<float>(sumEnergy / trackP);
    f.AvgHitEnergy          = static_cast<float>(meanE);
    f.SpreadPhi             = static_cast<float>(phiMax - phiMin);
    f.SpreadEta             = static_cast<float>(etaMax - etaMin);
    f.SpreadR               = static_cast<float>(Rmax - Rmin);
    f.MaxHitFrac            = static_cast<float>(maxHitE / sumEnergy);
    f.EnergyStdDev          = static_cast<float>(energyStdDev);
    f.EnergyConcentration   = static_cast<float>(energyConcentration);
    f.R_Disp                 = (n > 1) ? static_cast<float>(std::sqrt(sumR2diff / (n - 1))) : 0.f;
    f.R_DispWeighted         = static_cast<float>(std::sqrt(sumR2diffW / sumEnergy));
    f.Eta_DispWeighted       = static_cast<float>(std::sqrt(sumDEta2diffW / sumEnergy));
    f.Phi_DispWeighted       = static_cast<float>(std::sqrt(sumDPhi2diffW / sumEnergy));

    return f;
}

// Zbiera klastry z kolekcji TrackClusterMatch dopasowane do danego tracku.
template <typename MatchCollection>
void CollectMatchedClusters(const edm4eic::Track& trk,
                             const MatchCollection& matches,
                             std::vector<edm4eic::Cluster>& out)
{
    for (const auto& m : matches)
    {
        if (m.getTrack() == trk) out.push_back(m.getCluster());
    }
}

// =====================================================================
// Cechy ToF - odpowiednik ComputeToFFeatures z TestingMacro_Fixed.cxx,
// ale dzialajacy na kolekcjach edm4eic (zywy event) zamiast TTreeReaderArray.
// =====================================================================
struct ToFFeatures
{
    float Beta          = -999.f;
    float MassSq        = -999.f;
    float NHitsBarrel    = 0.f;
    float NHitsEndcap    = 0.f;
    float NHitsTotal     = 0.f;
    float MinDistBarrel  = -999.f;
    float MinDistEndcap  = -999.f;
    float AvgLenBarrel   = -999.f;
    float AvgLenEndcap   = -999.f;
    float HasToF         = 0.f;
};

// Kombinuje wiele hitow ToF w jeden estymator beta (liniowa regresja x = L/c vs t).
struct BetaEstimate { double beta; bool valid; };

static BetaEstimate CombineBetaHits(const std::vector<std::pair<double, double>>& hits)
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto& h : hits)
    {
        double t = h.first;
        double L = h.second;
        double x = L / c_light;
        sumXX += x * x;
        sumXT += x * t;
    }
    if (sumXX <= 0.0) return {0.0, false};
    double u_hat = sumXT / sumXX;
    if (u_hat <= 0.0) return {0.0, false};
    return {1.0 / u_hat, true};
}

template <typename HitCollection>
ToFFeatures ComputeToFFeatures(const TLorentzVector& Partic,
                                int charge,
                                const HitCollection& barrelHits,
                                const HitCollection& endcapHits)
{
    ToFFeatures f;

    double trackEta = Partic.Eta();
    double trackPhi = Partic.Phi();

    std::vector<std::pair<double, double>> matchedBarrel; // {time, length}
    std::vector<std::pair<double, double>> matchedEndcap;

    double sumLenBarrel = 0.0, minDistBarrel = 1e18;
    double sumLenEndcap = 0.0, minDistEndcap = 1e18;

    for (const auto& hit : barrelHits)
    {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta * dEta + dPhi * dPhi);

        if (dR < TOF_DR_CUT_BARREL && charge * dPhi < 0)
        {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (tof.distance_to_TOF < TOF_DIST_CUT_BARREL)
            {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedBarrel.push_back({hit.getTime(), length});
                sumLenBarrel += length;
                if (tof.distance_to_TOF < minDistBarrel) minDistBarrel = tof.distance_to_TOF;
            }
        }
    }

    for (const auto& hit : endcapHits)
    {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta * dEta + dPhi * dPhi);

        if (dR < TOF_DR_CUT_ENDCAP && charge * dPhi < 0)
        {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);
            if (tof.distance_to_TOF < TOF_DIST_CUT_ENDCAP)
            {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                matchedEndcap.push_back({hit.getTime(), length});
                sumLenEndcap += length;
                if (tof.distance_to_TOF < minDistEndcap) minDistEndcap = tof.distance_to_TOF;
            }
        }
    }

    f.NHitsBarrel = static_cast<float>(matchedBarrel.size());
    f.NHitsEndcap = static_cast<float>(matchedEndcap.size());
    f.NHitsTotal  = f.NHitsBarrel + f.NHitsEndcap;

    f.MinDistBarrel = matchedBarrel.empty() ? MISSING_SENTINEL : static_cast<float>(minDistBarrel);
    f.MinDistEndcap = matchedEndcap.empty() ? MISSING_SENTINEL : static_cast<float>(minDistEndcap);
    f.AvgLenBarrel  = matchedBarrel.empty() ? MISSING_SENTINEL : static_cast<float>(sumLenBarrel / matchedBarrel.size());
    f.AvgLenEndcap  = matchedEndcap.empty() ? MISSING_SENTINEL : static_cast<float>(sumLenEndcap / matchedEndcap.size());

    std::vector<std::pair<double, double>> allMatched;
    allMatched.insert(allMatched.end(), matchedBarrel.begin(), matchedBarrel.end());
    allMatched.insert(allMatched.end(), matchedEndcap.begin(), matchedEndcap.end());

    if (!allMatched.empty())
    {
        BetaEstimate be = CombineBetaHits(allMatched);
        if (be.valid)
        {
            double p = Partic.P();
            double msq = p * p * (1.0 / (be.beta * be.beta) - 1.0);
            f.Beta   = static_cast<float>(be.beta);
            f.MassSq = static_cast<float>(msq);
            f.HasToF = 1.f;
        }
    }

    return f;
}

// Wspolna funkcja uruchamiajaca dowolna sesje ONNX na dowolnym wektorze cech.
static double RunSession(Ort::Session& session, Ort::MemoryInfo& mem, const std::vector<float>& raw)
{
    int64_t shape[2] = {1, static_cast<int64_t>(raw.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(raw.data()), raw.size(), shape, 2);

    const char* input_names[]  = {"raw_features"};
    const char* output_names[] = {"probabilities"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    float* probs = output_tensors[0].GetTensorMutableData<float>();
    return probs[1]; // P(muon)
}

} // namespace MuonID_Detail


void InitMuonIDSessions()
{
    using namespace MuonID_Detail;

    if (g_ort_env) return; // juz zainicjalizowane

    g_ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MuonID");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // high-P: model calo-only, 30 cech (patrz build_raw_features w TestingMacro.cxx)
    g_session_highP = std::make_unique<Ort::Session>(*g_ort_env, "/run/media/epic/Data/Background/xgb_muonID_HE.onnx", session_options);
    // low-P: model calo(uproszczone)+ToF+kinematyka, 21 cech
    // (patrz build_raw_features_from_components w TestingMacro_Fixed.cxx)
    g_session_lowP  = std::make_unique<Ort::Session>(*g_ort_env, "/run/media/epic/Data/Background/xgb_muonID_LE.onnx", session_options);

    g_mem_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

double MuonID(edm4eic::ReconstructedParticle rcp, const podio::Frame* event)
{
    using namespace MuonID_Detail;

    InitMuonIDSessions();

    if (rcp.getTracks().empty()) return 0.0;
    const auto trk = rcp.getTracks().at(0);

    // --- edm4eic collections (HCal - jak w oryginale) ---
    const auto &hcalBarrelMatches  = event->get<edm4eic::TrackClusterMatchCollection>("HcalBarrelTrackClusterMatches");
    const auto &hcalEndcapNMatches = event->get<edm4eic::TrackClusterMatchCollection>("HcalEndcapNTrackClusterMatches");
    const auto &lfhcalMatches      = event->get<edm4eic::TrackClusterMatchCollection>("LFHCALTrackClusterMatches");

    // --- kinematyka ---
    const auto mom = rcp.getMomentum();
    TVector3 vec_mom(mom.x, mom.y, mom.z);

    constexpr double MUON_MASS_GEV = 0.1056583745;
    TLorentzVector Partic;
    Partic.SetVectM(vec_mom, MUON_MASS_GEV);

    double p    = vec_mom.Mag();
    double eta  = vec_mom.Eta();
    double phi  = vec_mom.Phi();

    // --- ECal: klastry juz dopasowane bezposrednio do czastki ---
    CaloFeatures ecalF = ComputeCaloFeaturesFromClusters(rcp.getClusters(), Partic, p);

    // --- HCal: zbieramy klastry z kolekcji dopasowan track-cluster ---
    std::vector<edm4eic::Cluster> hcalClusters;
    CollectMatchedClusters(trk, hcalBarrelMatches,  hcalClusters);
    CollectMatchedClusters(trk, hcalEndcapNMatches, hcalClusters);
    CollectMatchedClusters(trk, lfhcalMatches,      hcalClusters);

    CaloFeatures hcalF = ComputeCaloFeaturesFromClusters(hcalClusters, Partic, p);

    if (p >= P_THRESHOLD_GEV)
    {
        // ================= HIGH-P: model calo-only, 30 cech =================
        if (ecalF.Number <= 0 && hcalF.Number <= 0) return 0.0;

        std::vector<float> raw_highP = {
            ecalF.Energy,              hcalF.Energy,
            ecalF.Number,              hcalF.Number,
            ecalF.EoverP,              hcalF.EoverP,
            ecalF.AvgHitEnergy,        hcalF.AvgHitEnergy,
            ecalF.SpreadPhi,           ecalF.SpreadEta,           ecalF.SpreadR,
            hcalF.SpreadPhi,           hcalF.SpreadEta,           hcalF.SpreadR,
            ecalF.MaxHitFrac,          hcalF.MaxHitFrac,
            ecalF.EnergyStdDev,        hcalF.EnergyStdDev,
            ecalF.EnergyConcentration, hcalF.EnergyConcentration,
            ecalF.R_Disp,              ecalF.R_DispWeighted,
            ecalF.Eta_DispWeighted,    ecalF.Phi_DispWeighted,
            hcalF.R_Disp,              hcalF.R_DispWeighted,
            hcalF.Eta_DispWeighted,    hcalF.Phi_DispWeighted,
            static_cast<float>(p),     static_cast<float>(eta)
        };

        return RunSession(*g_session_highP, *g_mem_info, raw_highP);
    }
    else
    {
        // ================= LOW-P: model calo(uproszczone)+ToF, 21 cech =================
        // ZALOZENIE: nazwy/typy kolekcji ToF - PODMIEN jesli inne w Twoim EICrecon.
        const auto &tofBarrelHits = event->get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits");
        const auto &tofEndcapHits = event->get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits");

        int charge = static_cast<int>(rcp.getCharge());

        ToFFeatures tofF = ComputeToFFeatures(Partic, charge, tofBarrelHits, tofEndcapHits);

        if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5f) return 0.0;

        float ECalEoverP     = (ecalF.Number > 0) ? ecalF.EoverP     : MISSING_SENTINEL;
        float ECalMaxHitFrac = (ecalF.Number > 0) ? ecalF.MaxHitFrac : MISSING_SENTINEL;
        float HCalEoverP     = (hcalF.Number > 0) ? hcalF.EoverP     : MISSING_SENTINEL;
        float HCalMaxHitFrac = (hcalF.Number > 0) ? hcalF.MaxHitFrac : MISSING_SENTINEL;

        // Kolejnosc 1:1 z build_raw_features_from_components (TestingMacro_Fixed.cxx):
        std::vector<float> raw_lowP = {
            ecalF.Energy, ecalF.Number, ECalEoverP, ECalMaxHitFrac,
            hcalF.Energy, hcalF.Number, HCalEoverP, HCalMaxHitFrac,
            tofF.Beta, tofF.MassSq, tofF.NHitsBarrel, tofF.NHitsEndcap, tofF.NHitsTotal,
            tofF.MinDistBarrel, tofF.MinDistEndcap, tofF.AvgLenBarrel, tofF.AvgLenEndcap, tofF.HasToF,
            static_cast<float>(p), static_cast<float>(eta), static_cast<float>(phi)
        };

        return RunSession(*g_session_lowP, *g_mem_info, raw_lowP);
    }
}