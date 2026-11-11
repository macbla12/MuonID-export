// MuonID.cxx
//
// MuonID zgodne 1:1 z TestingMacro_MuonID_LowHigh.cxx:
//   p <= P_SPLIT_GEV -> xgb_muonID_LowP.onnx  (Calo + ToF, 41 cech)
//   p >  P_SPLIT_GEV -> xgb_muonID_HighP.onnx (Calo, 30 cech)
//
// Plik jest #include-owany jako zrodlo (example.cxx), wiec:
//   - NIE definiuje ExpandGlob ani main (jest w example),
//   - eksportuje: P_SPLIT_GEV oraz
//       float MuonID(const edm4eic::ReconstructedParticle &rcp,
//                    const podio::Frame *event, bool *hasDetectorInfo);
//
// hasDetectorInfo == false  <=>  w makrze testowym trak by zostal pominiety
// (`continue`): brak sladu/ToF/calo dla modelu Low, brak calo dla modelu High.

#pragma once

#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <onnxruntime_cxx_api.h>

#include <podio/Frame.h>

#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/TrackSegmentCollection.h>

using namespace std;

// =============================================================================
// Ustawienia (identyczne jak w makrze testowym)
// =============================================================================
static const double P_SPLIT_GEV = 0.7;

// Sciezki do modeli (mozna nadpisac zmiennymi srodowiskowymi)
static const char *MUONID_LOW_MODEL_DEFAULT  = "ONNX/xgb_muonID_LowP.onnx";
static const char *MUONID_HIGH_MODEL_DEFAULT = "ONNX/xgb_muonID_HighP.onnx";

namespace MuonIDImpl {

static const string SEG_COLL = "CentralTrackSegments";

static constexpr double c_light = 299.792458;   // [mm/ns]

// Korekta vz dla ToF: t_corr = t_hit - T0_PER_MM * vz
static const bool   CORRECT_T0   = true;
static const double T0_PER_MM    = -1.0 / c_light;
static const bool   CUT_BAD_PATH = true;
static const double L_TOL        = 10.0;   // [mm]

// Pomijanie hitow EcalBarrelImagingRecHits w ECal (jak w treningu LowP)
static const bool SKIP_IMAGING_HITS = true;

// Okna geometryczne ToF [mm]
static const double BTOF_RMIN = 600,  BTOF_RMAX = 720,  BTOF_ZMAX = 1200;
static const double FTOF_ZMIN = 1750, FTOF_ZMAX = 1950, FTOF_RMIN = 100, FTOF_RMAX = 650;

static const float MISSING_SENTINEL = -999.f;

// Parametry dopasowania ToF (jak w makrze)
static const double DR_CUT_BARREL   = 0.8;
static const double DR_CUT_ENDCAP   = 0.8;
static const double DIST_CUT_BARREL = 6.0;
static const double DIST_CUT_ENDCAP = 6.0;

// =============================================================================
// Narzedzia
// =============================================================================
inline bool HasCollection(const vector<string> &names, const string &n)
{
    return find(names.begin(), names.end(), n) != names.end();
}

inline bool InBarrelTOF(double x, double y, double z)
{
    const double R = std::hypot(x, y), az = std::fabs(z);
    return (R >= BTOF_RMIN && R <= BTOF_RMAX && az <= BTOF_ZMAX);
}

inline bool InEndcapTOF(double x, double y, double z)
{
    const double R = std::hypot(x, y), az = std::fabs(z);
    return (az >= FTOF_ZMIN && az <= FTOF_ZMAX && R >= FTOF_RMIN && R <= FTOF_RMAX);
}

// =============================================================================
// Kalorymetry
// =============================================================================
struct DetectorAssoc
{
    string clusterName;
    string matchCollName;
};

inline DetectorAssoc MakeDetectorAssoc(const string &clusterName)
{
    DetectorAssoc d;
    d.clusterName   = clusterName;
    d.matchCollName = clusterName.substr(0, clusterName.size() - string("Clusters").size()) + "TrackClusterMatches";
    return d;
}

struct TrackCalHits
{
    double sumEnergy = 0.0;
    double phiMin = 1e9, phiMax = -1e9;
    double etaMin = 1e9, etaMax = -1e9;
    double Rmin = 1e9, Rmax = -1e9;
    double maxHitE = -1.0;
    vector<double> R, dEta, dPhi, E;
};

// ECal: rcp.getClusters(); HCal: klastry z TrackClusterMatch dla tracka tej czastki.
inline TrackCalHits CollectHits(const edm4eic::ReconstructedParticle &rcp,
                                const vector<DetectorAssoc> &detectors,
                                const podio::Frame &frame,
                                const vector<string> &availNames,
                                bool fromRcpClusters,
                                const TLorentzVector &Partic)
{
    TrackCalHits result;
    vector<edm4eic::Cluster> clusters;
    set<podio::ObjectID> imagingHitIDs;

    if (fromRcpClusters)
    {
        for (const auto &c : rcp.getClusters()) clusters.push_back(c);

        if (SKIP_IMAGING_HITS && !clusters.empty())
        {
            if (HasCollection(availNames, "EcalBarrelImagingRecHits"))
            {
                for (const auto &ih : frame.get<edm4eic::CalorimeterHitCollection>("EcalBarrelImagingRecHits"))
                    imagingHitIDs.insert(ih.getObjectID());
            }
        }
    }
    else
    {
        if (rcp.getTracks().empty()) return result;
        const auto trk = rcp.getTracks().at(0);

        for (const auto &det : detectors)
        {
            if (!HasCollection(availNames, det.matchCollName)) continue;
            for (const auto &m : frame.get<edm4eic::TrackClusterMatchCollection>(det.matchCollName))
                if (m.getTrack() == trk) clusters.push_back(m.getCluster());
        }
    }

    for (const auto &cluster : clusters)
    {
        for (const auto &hit : cluster.getHits())
        {
            if (imagingHitIDs.count(hit.getObjectID())) continue;

            float HitE = hit.getEnergy();
            auto  pos  = hit.getPosition();

            TVector3 hitVec(pos.x, pos.y, pos.z);
            double hitEta = hitVec.Eta();
            double hitPhi = hitVec.Phi();
            double R = sqrt(pos.x * pos.x + pos.y * pos.y);

            double dEta = hitEta - Partic.Eta();
            double dPhi = TVector2::Phi_mpi_pi(hitPhi - Partic.Phi());

            result.sumEnergy += HitE;
            result.R.push_back(R);
            result.dEta.push_back(dEta);
            result.dPhi.push_back(dPhi);
            result.E.push_back(HitE);

            result.phiMin = min(result.phiMin, hitPhi); result.phiMax = max(result.phiMax, hitPhi);
            result.etaMin = min(result.etaMin, hitEta); result.etaMax = max(result.etaMax, hitEta);
            result.Rmin   = min(result.Rmin, R);        result.Rmax   = max(result.Rmax, R);

            if (HitE > result.maxHitE) result.maxHitE = HitE;
        }
    }
    return result;
}

struct TrackFeatures
{
    float Energy = 0.f, Number = 0.f, EoverP = 0.f, AvgHitEnergy = 0.f;
    float SpreadPhi = 0.f, SpreadEta = 0.f, SpreadR = 0.f, MaxHitFrac = 0.f;
    float EnergyStdDev = 0.f, EnergyConcentration = 0.f;
    float R_Disp = 0.f, R_DispWeighted = 0.f, Eta_DispWeighted = 0.f, Phi_DispWeighted = 0.f;
};

inline TrackFeatures ComputeTrackFeatures(const TrackCalHits &hits, double trackP)
{
    TrackFeatures f;
    if (hits.sumEnergy <= 0) return f;

    int n = static_cast<int>(hits.E.size());
    if (n == 0) return f;

    double sumEnergy = hits.sumEnergy;
    double meanE = sumEnergy / n;

    double sumSqDevE = 0.0, sumE2 = 0.0;
    for (double e : hits.E)
    {
        double devE = e - meanE;
        sumSqDevE += devE * devE;
        sumE2 += e * e;
    }
    double energyStdDev = sqrt(sumSqDevE / n);
    double energyConcentration = sumE2 / (sumEnergy * sumEnergy);

    double spreadPhi = acos(cos(hits.phiMax - hits.phiMin));
    double spreadEta = hits.etaMax - hits.etaMin;
    double spreadR   = hits.Rmax - hits.Rmin;

    double sumRw = 0.0;
    for (int k = 0; k < n; ++k) sumRw += hits.R[k] * hits.E[k];
    double meanR_w = sumRw / sumEnergy;

    double sumR2diff = 0.0, sumR2diffW = 0.0;
    double sumDEta2diffW = 0.0, sumDPhi2diffW = 0.0;
    for (int k = 0; k < n; ++k)
    {
        double dR = hits.R[k] - meanR_w;
        sumR2diff  += dR * dR;
        sumR2diffW += dR * dR * hits.E[k];

        sumDEta2diffW += hits.dEta[k] * hits.dEta[k] * hits.E[k];
        sumDPhi2diffW += hits.dPhi[k] * hits.dPhi[k] * hits.E[k];
    }

    double R_disp_unweighted = (n > 1) ? sqrt(sumR2diff / (n - 1)) : 0.0;
    double R_disp_weighted   = sqrt(sumR2diffW / sumEnergy);
    double Eta_disp_weighted = sqrt(sumDEta2diffW / sumEnergy);
    double Phi_disp_weighted = sqrt(sumDPhi2diffW / sumEnergy);

    f.Energy              = static_cast<float>(sumEnergy);
    f.Number              = static_cast<float>(n);
    f.EoverP              = static_cast<float>(sumEnergy / trackP);
    f.AvgHitEnergy        = static_cast<float>(meanE);
    f.SpreadPhi           = static_cast<float>(spreadPhi);
    f.SpreadEta           = static_cast<float>(spreadEta);
    f.SpreadR             = static_cast<float>(spreadR);
    f.MaxHitFrac          = static_cast<float>(hits.maxHitE / sumEnergy);
    f.EnergyStdDev        = static_cast<float>(energyStdDev);
    f.EnergyConcentration = static_cast<float>(energyConcentration);
    f.R_Disp              = static_cast<float>(R_disp_unweighted);
    f.R_DispWeighted      = static_cast<float>(R_disp_weighted);
    f.Eta_DispWeighted    = static_cast<float>(Eta_disp_weighted);
    f.Phi_DispWeighted    = static_cast<float>(Phi_disp_weighted);
    return f;
}

// =============================================================================
// ToF (tylko model Low)
// =============================================================================
struct BetaEstimate { double beta; bool valid; };

inline BetaEstimate CombineBeta(const std::vector<std::pair<double,double>> &hits)
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto &h : hits) {
        double t = h.first;
        double x = h.second / c_light;
        sumXX += x * x;
        sumXT += x * t;
    }
    if (sumXX <= 0.0) return {0.0, false};
    double u_hat = sumXT / sumXX;
    if (u_hat <= 0.0) return {0.0, false};
    return {1.0 / u_hat, true};
}

struct TrackToFFeatures
{
    float Beta          = -999.f;
    float MassSq        = -999.f;
    float NHitsBarrel   = 0.f;
    float NHitsEndcap   = 0.f;
    float NHitsTotal    = 0.f;
    float MinDistBarrel = -999.f;
    float MinDistEndcap = -999.f;
    float AvgLenBarrel  = -999.f;
    float AvgLenEndcap  = -999.f;
    float HasToF        = 0.f;
};

struct TofPoint { double x, y, z, path; };

inline void CollectSegmentTofPoints(const edm4eic::TrackSegmentCollection &segs,
                                    const edm4eic::Track &trk,
                                    const TVector3 &ref,
                                    vector<TofPoint> &barrelPts,
                                    vector<TofPoint> &endcapPts)
{
    for (const auto seg : segs) {
        if (seg.getTrack() != trk) continue;
        for (const auto pt : seg.getPoints()) {
            const double x = pt.position.x, y = pt.position.y, z = pt.position.z;
            if (CUT_BAD_PATH &&
                pt.pathlength < (TVector3(x, y, z) - ref).Mag() - L_TOL) continue;
            if (InBarrelTOF(x, y, z))      barrelPts.push_back({x, y, z, (double)pt.pathlength});
            else if (InEndcapTOF(x, y, z)) endcapPts.push_back({x, y, z, (double)pt.pathlength});
        }
    }
}

inline const TofPoint *NearestPoint(const vector<TofPoint> &pts, const TVector3 &hit, double &dist)
{
    const TofPoint *best = nullptr;
    double bestD = 1e18;
    for (const auto &p : pts) {
        double d = std::sqrt((p.x - hit.X())*(p.x - hit.X()) +
                             (p.y - hit.Y())*(p.y - hit.Y()) +
                             (p.z - hit.Z())*(p.z - hit.Z()));
        if (d < bestD) { bestD = d; best = &p; }
    }
    dist = bestD;
    return best;
}

inline TrackToFFeatures ComputeToFFeatures(
    const TLorentzVector &Partic,
    const edm4eic::Track &trk,
    const TVector3 &ref,
    const edm4eic::TrackSegmentCollection &segments,
    const edm4eic::TrackerHitCollection &barrelHits,
    const edm4eic::TrackerHitCollection &endcapHits,
    double dR_cut_barrel, double dR_cut_endcap,
    double dist_cut_barrel, double dist_cut_endcap)
{
    TrackToFFeatures f;

    const double vz     = ref.Z();
    const double tShift = CORRECT_T0 ? T0_PER_MM * vz : 0.0;

    vector<TofPoint> barrelPts, endcapPts;
    CollectSegmentTofPoints(segments, trk, ref, barrelPts, endcapPts);

    double trackEta = Partic.Eta();
    double trackPhi = Partic.Phi();

    std::vector<std::pair<double,double>> matchedBarrel, matchedEndcap; // {time, length}

    double sumLenBarrel = 0.0, minDistBarrel = 1e18;
    double sumLenEndcap = 0.0, minDistEndcap = 1e18;

    for (const auto &hit : barrelHits) {
        auto p = hit.getPosition();
        TVector3 ToFPos(p.x, p.y, p.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_barrel) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(barrelPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_barrel) {
                matchedBarrel.push_back({hit.getTime() - tShift, pt->path});
                sumLenBarrel += pt->path;
                if (dist < minDistBarrel) minDistBarrel = dist;
            }
        }
    }

    for (const auto &hit : endcapHits) {
        auto p = hit.getPosition();
        TVector3 ToFPos(p.x, p.y, p.z);
        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta*dEta + dPhi*dPhi);

        if (dR < dR_cut_endcap) {
            double dist = 0.0;
            const TofPoint *pt = NearestPoint(endcapPts, ToFPos, dist);
            if (!pt) continue;
            if (dist < dist_cut_endcap) {
                matchedEndcap.push_back({hit.getTime() - tShift, pt->path});
                sumLenEndcap += pt->path;
                if (dist < minDistEndcap) minDistEndcap = dist;
            }
        }
    }

    f.NHitsBarrel = (float)matchedBarrel.size();
    f.NHitsEndcap = (float)matchedEndcap.size();
    f.NHitsTotal  = f.NHitsBarrel + f.NHitsEndcap;

    f.MinDistBarrel = matchedBarrel.empty() ? -999.f : (float)minDistBarrel;
    f.MinDistEndcap = matchedEndcap.empty() ? -999.f : (float)minDistEndcap;
    f.AvgLenBarrel  = matchedBarrel.empty() ? -999.f : (float)(sumLenBarrel / matchedBarrel.size());
    f.AvgLenEndcap  = matchedEndcap.empty() ? -999.f : (float)(sumLenEndcap / matchedEndcap.size());

    std::vector<std::pair<double,double>> allMatched;
    allMatched.insert(allMatched.end(), matchedBarrel.begin(), matchedBarrel.end());
    allMatched.insert(allMatched.end(), matchedEndcap.begin(), matchedEndcap.end());

    if (!allMatched.empty()) {
        BetaEstimate be = CombineBeta(allMatched);
        if (be.valid) {
            double p = Partic.P();
            double msq = p*p * (1.0/(be.beta*be.beta) - 1.0);
            f.Beta   = (float)be.beta;
            f.MassSq = (float)msq;
            f.HasToF = 1.f;
        }
    }
    return f;
}

// =============================================================================
// ONNX - sesje tworzone leniwie przy pierwszym wywolaniu
// =============================================================================
struct OnnxModels
{
    Ort::Env            env;
    Ort::SessionOptions so;
    unique_ptr<Ort::Session> low;
    unique_ptr<Ort::Session> high;
    Ort::MemoryInfo     mem;

    static Ort::SessionOptions MakeOptions()
    {
        Ort::SessionOptions o;
        o.SetIntraOpNumThreads(1);
        o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        return o;
    }

    OnnxModels()
        : env(ORT_LOGGING_LEVEL_WARNING, "MuonID"),
          so(MakeOptions()),
          mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        const char *lowPath  = getenv("MUONID_LOW_MODEL")  ? getenv("MUONID_LOW_MODEL")  : MUONID_LOW_MODEL_DEFAULT;
        const char *highPath = getenv("MUONID_HIGH_MODEL") ? getenv("MUONID_HIGH_MODEL") : MUONID_HIGH_MODEL_DEFAULT;
        cout << "[MuonID] Low model : " << lowPath  << endl;
        cout << "[MuonID] High model: " << highPath << endl;
        low  = make_unique<Ort::Session>(env, lowPath,  so);
        high = make_unique<Ort::Session>(env, highPath, so);
    }
};

inline OnnxModels &Models()
{
    static OnnxModels m;   // inicjalizacja przy pierwszym uzyciu
    return m;
}

inline float RunSession(Ort::Session &session, Ort::MemoryInfo &mem, const std::vector<float> &raw)
{
    int64_t shape[2] = {1, static_cast<int64_t>(raw.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(raw.data()), raw.size(), shape, 2);

    const char* input_names[]  = {"raw_features"};
    const char* output_names[] = {"probabilities"};

    auto output = session.Run(Ort::RunOptions{nullptr},
                              input_names, &input_tensor, 1,
                              output_names, 1);

    float* probs = output[0].GetTensorMutableData<float>();
    return probs[1];   // P(muon)
}

// 41 cech (model LowP)
inline std::vector<float> BuildRawLow(const TrackFeatures &e, const TrackFeatures &h,
                                      const TrackToFFeatures &t,
                                      double trackP, double trackEta, double trackPhi)
{
    auto E = [&](float v) { return (e.Number > 0) ? v : MISSING_SENTINEL; };
    auto H = [&](float v) { return (h.Number > 0) ? v : MISSING_SENTINEL; };
    auto T = [&](float v) { return (t.HasToF > 0.5f) ? v : MISSING_SENTINEL; };

    return {
        e.Energy, e.Number, E(e.EoverP), E(e.AvgHitEnergy),
        E(e.SpreadPhi), E(e.SpreadEta), E(e.SpreadR), E(e.MaxHitFrac),
        E(e.EnergyStdDev), E(e.EnergyConcentration),
        E(e.R_Disp), E(e.R_DispWeighted), E(e.Eta_DispWeighted), E(e.Phi_DispWeighted),

        h.Energy, h.Number, H(h.EoverP), H(h.AvgHitEnergy),
        H(h.SpreadPhi), H(h.SpreadEta), H(h.SpreadR), H(h.MaxHitFrac),
        H(h.EnergyStdDev), H(h.EnergyConcentration),
        H(h.R_Disp), H(h.R_DispWeighted), H(h.Eta_DispWeighted), H(h.Phi_DispWeighted),

        T(t.Beta), T(t.MassSq), t.NHitsBarrel, t.NHitsEndcap, t.NHitsTotal,
        T(t.MinDistBarrel), T(t.MinDistEndcap), T(t.AvgLenBarrel), T(t.AvgLenEndcap), t.HasToF,

        static_cast<float>(trackP), static_cast<float>(trackEta), static_cast<float>(trackPhi)
    };
}

// 30 cech (model HighP) - bez sentineli
inline std::vector<float> BuildRawHigh(const TrackFeatures &e, const TrackFeatures &h,
                                       double trackP, double trackEta)
{
    return {
        e.Energy, h.Energy,
        e.Number, h.Number,
        e.EoverP, h.EoverP,
        e.AvgHitEnergy, h.AvgHitEnergy,
        e.SpreadPhi, e.SpreadEta, e.SpreadR,
        h.SpreadPhi, h.SpreadEta, h.SpreadR,
        e.MaxHitFrac, h.MaxHitFrac,
        e.EnergyStdDev, h.EnergyStdDev,
        e.EnergyConcentration, h.EnergyConcentration,
        e.R_Disp, e.R_DispWeighted,
        e.Eta_DispWeighted, e.Phi_DispWeighted,
        h.R_Disp, h.R_DispWeighted,
        h.Eta_DispWeighted, h.Phi_DispWeighted,
        static_cast<float>(trackP), static_cast<float>(trackEta)
    };
}

} // namespace MuonIDImpl

// =============================================================================
// MuonID - glowna funkcja
// =============================================================================
// Zwraca P(muon). Gdy hasDetectorInfo == false, trak nie moze byc ocenony
// (zwracane 0.f).
inline float MuonID(const edm4eic::ReconstructedParticle &rcp,
                    const podio::Frame *event,
                    bool *hasDetectorInfo = nullptr)
{
    using namespace MuonIDImpl;

    if (hasDetectorInfo) *hasDetectorInfo = false;
    if (!event) return 0.f;

    const podio::Frame &frame = *event;

    auto mom = rcp.getMomentum();
    TLorentzVector Partic;
    Partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());
    const double trackP   = Partic.P();
    const double trackEta = Partic.Eta();
    const double trackPhi = Partic.Phi();

    if (trackP <= 0) return 0.f;

    const bool useLow = (trackP <= P_SPLIT_GEV);

    // Dostepne kolekcje w tym evencie (czytane tylko to, co podano w readEntry)
    const vector<string> avail = frame.getAvailableCollections();

    static const vector<DetectorAssoc> hcalDetectors = {
        MakeDetectorAssoc("HcalBarrelClusters"),
        MakeDetectorAssoc("HcalEndcapNClusters"),
        MakeDetectorAssoc("LFHCALClusters")
    };
    static const vector<DetectorAssoc> ecalDetectors;   // ECal z rcp.getClusters()

    // ECal / HCal (wspolne)
    TrackCalHits ecalHits = CollectHits(rcp, ecalDetectors, frame, avail, true,  Partic);
    TrackCalHits hcalHits = CollectHits(rcp, hcalDetectors, frame, avail, false, Partic);
    TrackFeatures ecalF = ComputeTrackFeatures(ecalHits, trackP);
    TrackFeatures hcalF = ComputeTrackFeatures(hcalHits, trackP);

    auto &M = Models();
    float Pmu = 0.f;

    if (useLow)
    {
        // --- model LowP: wymaga tracka (HCal TCM + ToF) ---
        if (rcp.getTracks().empty()) return 0.f;
        const auto trk = rcp.getTracks().at(0);

        const auto rp = rcp.getReferencePoint();
        TVector3 refPoint(rp.x, rp.y, rp.z);

        static const edm4eic::TrackerHitCollection   emptyToF;
        static const edm4eic::TrackSegmentCollection emptySeg;

        const bool hasBToF = HasCollection(avail, "TOFBarrelSharedRecHits");
        const bool hasEToF = HasCollection(avail, "TOFEndcapSharedRecHits");
        const bool hasSeg  = HasCollection(avail, SEG_COLL);

        const auto &barrelToF = hasBToF ? frame.get<edm4eic::TrackerHitCollection>("TOFBarrelSharedRecHits") : emptyToF;
        const auto &endcapToF = hasEToF ? frame.get<edm4eic::TrackerHitCollection>("TOFEndcapSharedRecHits") : emptyToF;
        const auto &segments  = hasSeg  ? frame.get<edm4eic::TrackSegmentCollection>(SEG_COLL)             : emptySeg;

        TrackToFFeatures tofF = ComputeToFFeatures(
            Partic, trk, refPoint, segments, barrelToF, endcapToF,
            DR_CUT_BARREL, DR_CUT_ENDCAP, DIST_CUT_BARREL, DIST_CUT_ENDCAP);

        if (ecalF.Number <= 0 && hcalF.Number <= 0 && tofF.HasToF < 0.5) return 0.f;

        std::vector<float> raw = BuildRawLow(ecalF, hcalF, tofF, trackP, trackEta, trackPhi);
        Pmu = RunSession(*M.low, M.mem, raw);
    }
    else
    {
        // --- model HighP: tylko Calo ---
        if (ecalF.Number <= 0 && hcalF.Number <= 0) return 0.f;

        std::vector<float> raw = BuildRawHigh(ecalF, hcalF, trackP, trackEta);
        Pmu = RunSession(*M.high, M.mem, raw);
    }

    if (hasDetectorInfo) *hasDetectorInfo = true;
    return Pmu;
}