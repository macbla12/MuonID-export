// MuonID.cxx
//
// Combined MuonID:
//   track p < P_SPLIT_GEV  -> LowP model: full Calo shower-shape + ToF
//   track p >= P_SPLIT_GEV -> HighP model: detailed Calo
//
// Public interface:
//
//   float MuonID(const edm4eic::ReconstructedParticle& rcp,
//                const podio::Frame* event,
//                bool* hasDetectorInfo = nullptr);
//
// Returns:
//   P(muon)
//
// hasDetectorInfo (optional out-param):
//   Set to true only if the track left enough of a signal in the
//   calorimeters/ToF to actually run the model on it (the same "keep"
//   condition used by TestingMacro_Combined_Podio.cxx to decide whether a
//   track counts toward "Total" there). Set to false in every other case
//   (no event, trackP<=0, or no usable detector info at all) -- in all of
//   those cases the returned P(muon) is a meaningless 0.0f, NOT a genuine
//   "identified as not-muon" score, and callers that want a proper
//   "found & capable of ID" vs "correctly identified" efficiency (as
//   opposed to a single "identified / reconstructed" number) need this
//   flag to tell the two situations apart.
//
// The function itself does NOT:
//   - perform truth selection
//   - apply a MuonID cut
//   - fill histograms
//   - run over events
//   - print performance numbers
//
// Those things belong in the testing macro (example()).

#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TMath.h>

#include <onnxruntime_cxx_api.h>

#include <podio/Frame.h>

#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackerHitCollection.h>

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "ToFSim.cxx"


// ============================================================================
// Configuration
// ============================================================================

static constexpr double P_SPLIT_GEV = 1.0;

// Change these paths if necessary.
// NOTE: LOWP_ONNX_PATH must point at a model retrained on the FULL
// 41-feature RAW_COLS layout (see TrainMuonID.py / CombinedCaloToFAnalysis.cxx),
// not the old 21-feature simple-calo-features model.
static const char* LOWP_ONNX_PATH =
    "../ToF/ONNX/xgb_muonID.onnx";

static const char* HIGHP_ONNX_PATH =
    "../CalorimetryHits/ONNX/xgb_muonID.onnx";

static constexpr double TIMING_CUT_NS = 20.0;

static constexpr float MISSING_SENTINEL = -999.f;


// ============================================================================
// Detector association helper
// ============================================================================

struct DetectorAssoc
{
    std::string clusterName;
    std::string assocCollName;
};


static DetectorAssoc MakeDetectorAssoc(const std::string& clusterName)
{
    DetectorAssoc d;

    d.clusterName = clusterName;

    // e.g.
    // EcalBarrelImagingClusters
    //              ->
    // EcalBarrelImagingClusterAssociations
    d.assocCollName =
        clusterName.substr(0, clusterName.size() - 1)
        + "Associations";

    return d;
}


// ============================================================================
// ONNX inference
// ============================================================================

static float run_muon_id(
    Ort::Session& session,
    Ort::MemoryInfo& memoryInfo,
    const std::vector<float>& raw)
{
    int64_t shape[2] = {
        1,
        static_cast<int64_t>(raw.size())
    };

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float*>(raw.data()),
            raw.size(),
            shape,
            2
        );

    const char* inputNames[] = {"raw_features"};
    const char* outputNames[] = {"probabilities"};

    auto output = session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    float* probs =
        output[0].GetTensorMutableData<float>();

    // probabilities = [P(pion), P(muon)]
    return probs[1];
}


// ============================================================================
// SHARED CALO FEATURES (full shower-shape set)
//
// Used by BOTH the HighP path (29-feature model, no ToF) and the LowP path
// (41-feature model, full calo + ToF). Kept the "HighP_*" names since the
// computation itself is identical for both regimes -- only what gets built
// into the final raw-feature vector differs.
// ============================================================================

struct HighP_TrackCalHits
{
    double sumEnergy = 0.0;

    double phiMin = 1e9;
    double phiMax = -1e9;

    double etaMin = 1e9;
    double etaMax = -1e9;

    double Rmin = 1e9;
    double Rmax = -1e9;

    double maxHitE = -1.0;

    std::vector<double> R;
    std::vector<double> dEta;
    std::vector<double> dPhi;
    std::vector<double> E;
};


static HighP_TrackCalHits HighP_CollectHits(
    const edm4hep::MCParticle& simPart,
    const std::vector<DetectorAssoc>& detectors,
    const podio::Frame& frame,
    const TLorentzVector& particle,
    double timingCutNs)
{
    HighP_TrackCalHits result;

    for (const auto& det : detectors)
    {
        const auto& assocColl =
            frame.get<
                edm4eic::MCRecoClusterParticleAssociationCollection
            >(det.assocCollName);

        for (const auto& assoc : assocColl)
        {
            if (assoc.getSim() != simPart)
                continue;

            const auto cluster = assoc.getRec();

            for (const auto& hit : cluster.getHits())
            {
                const float hitE = hit.getEnergy();
                const float hitT = hit.getTime();

                if (hitT > timingCutNs)
                    continue;

                const auto pos = hit.getPosition();

                TVector3 hitVec(
                    pos.x,
                    pos.y,
                    pos.z
                );

                const double hitEta = hitVec.Eta();
                const double hitPhi = hitVec.Phi();

                const double R =
                    std::sqrt(
                        pos.x * pos.x +
                        pos.y * pos.y
                    );

                const double dEta =
                    hitEta - particle.Eta();

                const double dPhi =
                    TVector2::Phi_mpi_pi(
                        hitPhi - particle.Phi()
                    );

                result.sumEnergy += hitE;

                result.R.push_back(R);
                result.dEta.push_back(dEta);
                result.dPhi.push_back(dPhi);
                result.E.push_back(hitE);

                result.phiMin =
                    std::min(result.phiMin, hitPhi);

                result.phiMax =
                    std::max(result.phiMax, hitPhi);

                result.etaMin =
                    std::min(result.etaMin, hitEta);

                result.etaMax =
                    std::max(result.etaMax, hitEta);

                result.Rmin =
                    std::min(result.Rmin, R);

                result.Rmax =
                    std::max(result.Rmax, R);

                if (hitE > result.maxHitE)
                    result.maxHitE = hitE;
            }
        }
    }

    return result;
}


struct HighP_TrackFeatures
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


static HighP_TrackFeatures HighP_ComputeTrackFeatures(
    const HighP_TrackCalHits& hits,
    double trackP)
{
    HighP_TrackFeatures f;

    if (hits.sumEnergy <= 0.0)
        return f;

    const int n =
        static_cast<int>(hits.E.size());

    if (n == 0)
        return f;

    const double sumEnergy = hits.sumEnergy;

    const double meanE =
        sumEnergy / n;

    double sumSqDevE = 0.0;
    double sumE2 = 0.0;

    for (double e : hits.E)
    {
        const double devE =
            e - meanE;

        sumSqDevE += devE * devE;
        sumE2 += e * e;
    }

    const double energyStdDev =
        std::sqrt(sumSqDevE / n);

    const double energyConcentration =
        sumE2 / (sumEnergy * sumEnergy);

    const double spreadPhi =
        hits.phiMax - hits.phiMin;

    const double spreadEta =
        hits.etaMax - hits.etaMin;

    const double spreadR =
        hits.Rmax - hits.Rmin;


    double sumRw = 0.0;

    for (int k = 0; k < n; ++k)
    {
        sumRw +=
            hits.R[k] * hits.E[k];
    }

    const double meanR_w =
        sumRw / sumEnergy;


    double sumR2diff = 0.0;
    double sumR2diffW = 0.0;
    double sumDEta2diffW = 0.0;
    double sumDPhi2diffW = 0.0;

    for (int k = 0; k < n; ++k)
    {
        const double dR =
            hits.R[k] - meanR_w;

        sumR2diff +=
            dR * dR;

        sumR2diffW +=
            dR * dR * hits.E[k];

        sumDEta2diffW +=
            hits.dEta[k] *
            hits.dEta[k] *
            hits.E[k];

        sumDPhi2diffW +=
            hits.dPhi[k] *
            hits.dPhi[k] *
            hits.E[k];
    }

    const double R_disp_unweighted =
        (n > 1)
            ? std::sqrt(
                  sumR2diff / (n - 1)
              )
            : 0.0;

    const double R_disp_weighted =
        std::sqrt(
            sumR2diffW / sumEnergy
        );

    const double Eta_disp_weighted =
        std::sqrt(
            sumDEta2diffW / sumEnergy
        );

    const double Phi_disp_weighted =
        std::sqrt(
            sumDPhi2diffW / sumEnergy
        );


    f.Energy =
        static_cast<float>(sumEnergy);

    f.Number =
        static_cast<float>(n);

    f.EoverP =
        static_cast<float>(
            sumEnergy / trackP
        );

    f.AvgHitEnergy =
        static_cast<float>(meanE);

    f.SpreadPhi =
        static_cast<float>(spreadPhi);

    f.SpreadEta =
        static_cast<float>(spreadEta);

    f.SpreadR =
        static_cast<float>(spreadR);

    f.MaxHitFrac =
        static_cast<float>(
            hits.maxHitE / sumEnergy
        );

    f.EnergyStdDev =
        static_cast<float>(energyStdDev);

    f.EnergyConcentration =
        static_cast<float>(
            energyConcentration
        );

    f.R_Disp =
        static_cast<float>(
            R_disp_unweighted
        );

    f.R_DispWeighted =
        static_cast<float>(
            R_disp_weighted
        );

    f.Eta_DispWeighted =
        static_cast<float>(
            Eta_disp_weighted
        );

    f.Phi_DispWeighted =
        static_cast<float>(
            Phi_disp_weighted
        );

    return f;
}


static std::vector<float> HighP_build_raw_features(
    float ECalEnergy,
    float HCalEnergy,
    float ECalNumber,
    float HCalNumber,
    float ECalEoverP,
    float HCalEoverP,
    float ECalAvgHitEnergy,
    float HCalAvgHitEnergy,
    float ECalSpreadPhi,
    float ECalSpreadEta,
    float ECalSpreadR,
    float HCalSpreadPhi,
    float HCalSpreadEta,
    float HCalSpreadR,
    float ECalMaxHitFrac,
    float HCalMaxHitFrac,
    float ECalEnergyStdDev,
    float HCalEnergyStdDev,
    float ECalEnergyConcentration,
    float HCalEnergyConcentration,
    float ECalR_Disp,
    float ECalR_DispWeighted,
    float ECalEta_DispWeighted,
    float ECalPhi_DispWeighted,
    float HCalR_Disp,
    float HCalR_DispWeighted,
    float HCalEta_DispWeighted,
    float HCalPhi_DispWeighted,
    float TrackMomentum,
    float TrackEta)
{
    return {
        ECalEnergy,
        HCalEnergy,

        ECalNumber,
        HCalNumber,

        ECalEoverP,
        HCalEoverP,

        ECalAvgHitEnergy,
        HCalAvgHitEnergy,

        ECalSpreadPhi,
        ECalSpreadEta,
        ECalSpreadR,

        HCalSpreadPhi,
        HCalSpreadEta,
        HCalSpreadR,

        ECalMaxHitFrac,
        HCalMaxHitFrac,

        ECalEnergyStdDev,
        HCalEnergyStdDev,

        ECalEnergyConcentration,
        HCalEnergyConcentration,

        ECalR_Disp,
        ECalR_DispWeighted,
        ECalEta_DispWeighted,
        ECalPhi_DispWeighted,

        HCalR_Disp,
        HCalR_DispWeighted,
        HCalEta_DispWeighted,
        HCalPhi_DispWeighted,

        TrackMomentum,
        TrackEta
    };
}


// ============================================================================
// ToF
// ============================================================================

struct BetaEstimate
{
    double beta;
    bool valid;
};


static BetaEstimate CombineBeta(
    const std::vector<
        std::pair<double, double>
    >& hits)
{
    double sumXX = 0.0;
    double sumXT = 0.0;

    for (const auto& h : hits)
    {
        const double t = h.first;
        const double L = h.second;

        const double x =
            L / c_light;

        sumXX += x * x;
        sumXT += x * t;
    }

    if (sumXX <= 0.0)
        return {0.0, false};

    const double u_hat =
        sumXT / sumXX;

    if (u_hat <= 0.0)
        return {0.0, false};

    return {
        1.0 / u_hat,
        true
    };
}


struct LowP_TrackToFFeatures
{
    float Beta = -999.f;
    float MassSq = -999.f;

    float NHitsBarrel = 0.f;
    float NHitsEndcap = 0.f;
    float NHitsTotal = 0.f;

    float MinDistBarrel = -999.f;
    float MinDistEndcap = -999.f;

    float AvgLenBarrel = -999.f;
    float AvgLenEndcap = -999.f;

    float HasToF = 0.f;
};


template <typename HitCollection>
static LowP_TrackToFFeatures LowP_ComputeToFFeatures(
    const TLorentzVector& particle,
    int charge,
    const HitCollection& barrelHits,
    const HitCollection& endcapHits,
    double dR_cut_barrel,
    double dR_cut_endcap,
    double dist_cut_barrel,
    double dist_cut_endcap)
{
    LowP_TrackToFFeatures f;

    const double trackEta =
        particle.Eta();

    const double trackPhi =
        particle.Phi();


    std::vector<
        std::pair<double, double>
    > matchedBarrel;

    std::vector<
        std::pair<double, double>
    > matchedEndcap;


    double sumLenBarrel = 0.0;
    double minDistBarrel = 1e18;

    double sumLenEndcap = 0.0;
    double minDistEndcap = 1e18;


    // ------------------------------------------------------------------------
    // Barrel
    // ------------------------------------------------------------------------

    for (const auto& hit : barrelHits)
    {
        const auto pos =
            hit.getPosition();

        TVector3 tofPos(
            pos.x,
            pos.y,
            pos.z
        );

        const double dEta =
            trackEta - tofPos.Eta();

        const double dPhi =
            TVector2::Phi_mpi_pi(
                trackPhi - tofPos.Phi()
            );

        const double dR =
            std::sqrt(
                dEta * dEta +
                dPhi * dPhi
            );


        if (dR < dR_cut_barrel &&
            charge * dPhi < 0)
        {
            ToFResults tof =
                ToFSim(
                    particle,
                    charge,
                    tofPos
                );

            if (tof.distance_to_TOF <
                dist_cut_barrel)
            {
                const double length =
                    tof.DistanceCheck
                        ? tof.track_length
                        : tofPos.Mag();

                matchedBarrel.push_back(
                    {
                        hit.getTime(),
                        length
                    }
                );

                sumLenBarrel += length;

                if (tof.distance_to_TOF <
                    minDistBarrel)
                {
                    minDistBarrel =
                        tof.distance_to_TOF;
                }
            }
        }
    }


    // ------------------------------------------------------------------------
    // Endcap
    // ------------------------------------------------------------------------

    for (const auto& hit : endcapHits)
    {
        const auto pos =
            hit.getPosition();

        TVector3 tofPos(
            pos.x,
            pos.y,
            pos.z
        );

        const double dEta =
            trackEta - tofPos.Eta();

        const double dPhi =
            TVector2::Phi_mpi_pi(
                trackPhi - tofPos.Phi()
            );

        const double dR =
            std::sqrt(
                dEta * dEta +
                dPhi * dPhi
            );


        if (dR < dR_cut_endcap &&
            charge * dPhi < 0)
        {
            ToFResults tof =
                ToFSim(
                    particle,
                    charge,
                    tofPos
                );

            if (tof.distance_to_TOF <
                dist_cut_endcap)
            {
                const double length =
                    tof.DistanceCheck
                        ? tof.track_length
                        : tofPos.Mag();

                matchedEndcap.push_back(
                    {
                        hit.getTime(),
                        length
                    }
                );

                sumLenEndcap += length;

                if (tof.distance_to_TOF <
                    minDistEndcap)
                {
                    minDistEndcap =
                        tof.distance_to_TOF;
                }
            }
        }
    }


    // ------------------------------------------------------------------------
    // Basic ToF quantities
    // ------------------------------------------------------------------------

    f.NHitsBarrel =
        static_cast<float>(
            matchedBarrel.size()
        );

    f.NHitsEndcap =
        static_cast<float>(
            matchedEndcap.size()
        );

    f.NHitsTotal =
        f.NHitsBarrel +
        f.NHitsEndcap;


    f.MinDistBarrel =
        matchedBarrel.empty()
            ? -999.f
            : static_cast<float>(
                  minDistBarrel
              );

    f.MinDistEndcap =
        matchedEndcap.empty()
            ? -999.f
            : static_cast<float>(
                  minDistEndcap
              );


    f.AvgLenBarrel =
        matchedBarrel.empty()
            ? -999.f
            : static_cast<float>(
                  sumLenBarrel /
                  matchedBarrel.size()
              );

    f.AvgLenEndcap =
        matchedEndcap.empty()
            ? -999.f
            : static_cast<float>(
                  sumLenEndcap /
                  matchedEndcap.size()
              );


    // ------------------------------------------------------------------------
    // Beta and mass^2
    // ------------------------------------------------------------------------

    std::vector<
        std::pair<double, double>
    > allMatched;

    allMatched.insert(
        allMatched.end(),
        matchedBarrel.begin(),
        matchedBarrel.end()
    );

    allMatched.insert(
        allMatched.end(),
        matchedEndcap.begin(),
        matchedEndcap.end()
    );


    if (!allMatched.empty())
    {
        const BetaEstimate be =
            CombineBeta(allMatched);

        if (be.valid)
        {
            const double p =
                particle.P();

            const double massSq =
                p * p *
                (
                    1.0 /
                    (be.beta * be.beta)
                    - 1.0
                );

            f.Beta =
                static_cast<float>(
                    be.beta
                );

            f.MassSq =
                static_cast<float>(
                    massSq
                );

            f.HasToF = 1.f;
        }
    }

    return f;
}


// ============================================================================
// LOW-P raw-feature vector: full calo shower-shape (same computation as
// HighP, via HighP_CollectHits / HighP_ComputeTrackFeatures) for BOTH ECal
// and HCal, plus ToF, plus track kinematics. 41 values, in exactly the
// RAW_COLS order used by CombinedCaloToFAnalysis.cxx / TrainMuonID.py:
//
//   ECal: Energy, Number, EoverP, AvgHitEnergy, SpreadPhi, SpreadEta,
//         SpreadR, MaxHitFrac, EnergyStdDev, EnergyConcentration, R_Disp,
//         R_DispWeighted, Eta_DispWeighted, Phi_DispWeighted   (14)
//   HCal: same 14 fields
//   ToF:  Beta, MassSq, NHitsBarrel, NHitsEndcap, NHitsTotal, MinDistBarrel,
//         MinDistEndcap, AvgLenBarrel, AvgLenEndcap, HasToF   (10)
//   Track: Momentum, Eta, Phi   (3)
//
// Sentinel convention (must match the training pipeline exactly): Energy
// and Number stay real (0 when a subsystem had no hits); every other calo
// shower-shape field becomes MISSING_SENTINEL when Number<=0 for that
// subsystem. The ONNX preprocessing node does the sentinel imputation
// internally, so we must pass the raw sentinel through, not pre-impute it.
// ============================================================================

struct LowP_CaloRaw
{
    float Energy, Number, EoverP, AvgHitEnergy, SpreadPhi, SpreadEta, SpreadR,
          MaxHitFrac, EnergyStdDev, EnergyConcentration, R_Disp, R_DispWeighted,
          Eta_DispWeighted, Phi_DispWeighted;
};

static LowP_CaloRaw LowP_ApplySentinel(const HighP_TrackFeatures& f)
{
    const bool hasHits = f.Number > 0.f;
    LowP_CaloRaw r;
    r.Energy               = f.Energy;
    r.Number                = f.Number;
    r.EoverP                = hasHits ? f.EoverP : MISSING_SENTINEL;
    r.AvgHitEnergy           = hasHits ? f.AvgHitEnergy : MISSING_SENTINEL;
    r.SpreadPhi              = hasHits ? f.SpreadPhi : MISSING_SENTINEL;
    r.SpreadEta              = hasHits ? f.SpreadEta : MISSING_SENTINEL;
    r.SpreadR                = hasHits ? f.SpreadR : MISSING_SENTINEL;
    r.MaxHitFrac             = hasHits ? f.MaxHitFrac : MISSING_SENTINEL;
    r.EnergyStdDev           = hasHits ? f.EnergyStdDev : MISSING_SENTINEL;
    r.EnergyConcentration    = hasHits ? f.EnergyConcentration : MISSING_SENTINEL;
    r.R_Disp                 = hasHits ? f.R_Disp : MISSING_SENTINEL;
    r.R_DispWeighted          = hasHits ? f.R_DispWeighted : MISSING_SENTINEL;
    r.Eta_DispWeighted        = hasHits ? f.Eta_DispWeighted : MISSING_SENTINEL;
    r.Phi_DispWeighted        = hasHits ? f.Phi_DispWeighted : MISSING_SENTINEL;
    return r;
}

static std::vector<float> LowP_build_raw_features(
    float ECalEnergy, float ECalNumber, float ECalEoverP, float ECalAvgHitEnergy,
    float ECalSpreadPhi, float ECalSpreadEta, float ECalSpreadR, float ECalMaxHitFrac,
    float ECalEnergyStdDev, float ECalEnergyConcentration,
    float ECalR_Disp, float ECalR_DispWeighted, float ECalEta_DispWeighted, float ECalPhi_DispWeighted,

    float HCalEnergy, float HCalNumber, float HCalEoverP, float HCalAvgHitEnergy,
    float HCalSpreadPhi, float HCalSpreadEta, float HCalSpreadR, float HCalMaxHitFrac,
    float HCalEnergyStdDev, float HCalEnergyConcentration,
    float HCalR_Disp, float HCalR_DispWeighted, float HCalEta_DispWeighted, float HCalPhi_DispWeighted,

    float ToFBeta, float ToFMassSq, float ToFNHitsBarrel, float ToFNHitsEndcap, float ToFNHitsTotal,
    float ToFMinDistBarrel, float ToFMinDistEndcap, float ToFAvgLenBarrel, float ToFAvgLenEndcap, float ToFHasToF,

    float TrackMomentum, float TrackEta, float TrackPhi)
{
    return {
        ECalEnergy, ECalNumber, ECalEoverP, ECalAvgHitEnergy,
        ECalSpreadPhi, ECalSpreadEta, ECalSpreadR, ECalMaxHitFrac,
        ECalEnergyStdDev, ECalEnergyConcentration,
        ECalR_Disp, ECalR_DispWeighted, ECalEta_DispWeighted, ECalPhi_DispWeighted,

        HCalEnergy, HCalNumber, HCalEoverP, HCalAvgHitEnergy,
        HCalSpreadPhi, HCalSpreadEta, HCalSpreadR, HCalMaxHitFrac,
        HCalEnergyStdDev, HCalEnergyConcentration,
        HCalR_Disp, HCalR_DispWeighted, HCalEta_DispWeighted, HCalPhi_DispWeighted,

        ToFBeta, ToFMassSq, ToFNHitsBarrel, ToFNHitsEndcap, ToFNHitsTotal,
        ToFMinDistBarrel, ToFMinDistEndcap, ToFAvgLenBarrel, ToFAvgLenEndcap, ToFHasToF,

        TrackMomentum, TrackEta, TrackPhi
    };
}


// ============================================================================
// Find the MC particle associated with this reconstructed particle
//
// This is needed because the original testing macro obtained simPart from
//
//   trackAssocs[particle].getSim()
//
// but MuonID itself receives only:
//
//   (rcp, event)
//
// Therefore the association lookup has to happen inside MuonID.
//
// ============================================================================

static edm4hep::MCParticle FindSimParticle(
    const edm4eic::ReconstructedParticle& rcp,
    const podio::Frame& frame)
{
    const auto& associations =
        frame.get<
            edm4eic::MCRecoParticleAssociationCollection
        >("ReconstructedChargedParticleAssociations");

    for (const auto& assoc : associations)
    {
        if (assoc.getRec() == rcp)
        {
            return assoc.getSim();
        }
    }

    // Invalid MCParticle handle if no association was found.
    return edm4hep::MCParticle();
}


// ============================================================================
// Main MuonID function
// ============================================================================

float MuonID(
    const edm4eic::ReconstructedParticle& rcp,
    const podio::Frame* event,
    bool* hasDetectorInfo = nullptr)
{
    // ------------------------------------------------------------------------
    // Default: "no usable info" until proven otherwise below.
    // ------------------------------------------------------------------------

    if (hasDetectorInfo != nullptr)
        *hasDetectorInfo = false;


    // ------------------------------------------------------------------------
    // Basic validity check
    // ------------------------------------------------------------------------

    if (event == nullptr)
        return 0.0f;


    // ------------------------------------------------------------------------
    // Reconstructed particle kinematics
    // ------------------------------------------------------------------------

    const auto mom =
        rcp.getMomentum();

    TLorentzVector particle;

    particle.SetPxPyPzE(
        mom.x,
        mom.y,
        mom.z,
        rcp.getEnergy()
    );


    const double trackP =
        particle.P();

    const double trackEta =
        particle.Eta();

    const double trackPhi =
        particle.Phi();


    if (trackP <= 0.0)
        return 0.0f;


    // ------------------------------------------------------------------------
    // MC association
    //
    // The current feature implementation uses the MC particle to find
    // calorimeter clusters associated with the track.
    // ------------------------------------------------------------------------

    const edm4hep::MCParticle simPart =
        FindSimParticle(
            rcp,
            *event
        );




    // ------------------------------------------------------------------------
    // Detector collections
    // ------------------------------------------------------------------------

    const std::vector<DetectorAssoc> ecalDetectors = {
        MakeDetectorAssoc(
            "EcalBarrelImagingClusters"
        ),
        MakeDetectorAssoc(
            "EcalBarrelScFiClusters"
        ),
        MakeDetectorAssoc(
            "EcalEndcapPClusters"
        ),
        MakeDetectorAssoc(
            "EcalEndcapNClusters"
        )
    };


    const std::vector<DetectorAssoc> hcalDetectors = {
        MakeDetectorAssoc(
            "HcalBarrelClusters"
        ),
        MakeDetectorAssoc(
            "HcalEndcapNClusters"
        ),
        MakeDetectorAssoc(
            "LFHCALClusters"
        )
    };


    // =========================================================================
    // LOW-P
    // =========================================================================

    if (trackP < P_SPLIT_GEV)
    {
        // ---------------------------------------------------------------------
        // Get ToF collections
        // ---------------------------------------------------------------------

        const auto& tofBarrelHits =
            event->get<
                edm4eic::TrackerHitCollection
            >("TOFBarrelRecHits");

        const auto& tofEndcapHits =
            event->get<
                edm4eic::TrackerHitCollection
            >("TOFEndcapRecHits");


        // ---------------------------------------------------------------------
        // ECal (full shower-shape set, same computation as HighP)
        // ---------------------------------------------------------------------

        const HighP_TrackCalHits ecalHits =
            HighP_CollectHits(
                simPart,
                ecalDetectors,
                *event,
                particle,
                TIMING_CUT_NS
            );

        const HighP_TrackFeatures ecalF =
            HighP_ComputeTrackFeatures(
                ecalHits,
                trackP
            );


        // ---------------------------------------------------------------------
        // HCal (full shower-shape set, same computation as HighP)
        // ---------------------------------------------------------------------

        const HighP_TrackCalHits hcalHits =
            HighP_CollectHits(
                simPart,
                hcalDetectors,
                *event,
                particle,
                TIMING_CUT_NS
            );

        const HighP_TrackFeatures hcalF =
            HighP_ComputeTrackFeatures(
                hcalHits,
                trackP
            );


        // ---------------------------------------------------------------------
        // ToF
        // ---------------------------------------------------------------------

        const int charge =
            static_cast<int>(
                rcp.getCharge()
            );

        const LowP_TrackToFFeatures tofF =
            LowP_ComputeToFFeatures(
                particle,
                charge,
                tofBarrelHits,
                tofEndcapHits,
                0.8,   // dR barrel
                0.8,   // dR endcap
                6.0,   // distance barrel
                6.0    // distance endcap
            );


        // ---------------------------------------------------------------------
        // No useful detector information -> not "found" for ID purposes.
        // ---------------------------------------------------------------------

        if (ecalF.Number <= 0.0f &&
            hcalF.Number <= 0.0f &&
            tofF.HasToF < 0.5f)
        {
            return 0.0f;
        }

        if (hasDetectorInfo != nullptr)
            *hasDetectorInfo = true;


        // ---------------------------------------------------------------------
        // Missing-value treatment (sentinel), full calo shower-shape set.
        // ---------------------------------------------------------------------

        const LowP_CaloRaw ecalR = LowP_ApplySentinel(ecalF);
        const LowP_CaloRaw hcalR = LowP_ApplySentinel(hcalF);


        // ---------------------------------------------------------------------
        // 41-feature LowP vector
        // ---------------------------------------------------------------------

        const std::vector<float> raw =
            LowP_build_raw_features(
                ecalR.Energy, ecalR.Number, ecalR.EoverP, ecalR.AvgHitEnergy,
                ecalR.SpreadPhi, ecalR.SpreadEta, ecalR.SpreadR, ecalR.MaxHitFrac,
                ecalR.EnergyStdDev, ecalR.EnergyConcentration,
                ecalR.R_Disp, ecalR.R_DispWeighted, ecalR.Eta_DispWeighted, ecalR.Phi_DispWeighted,

                hcalR.Energy, hcalR.Number, hcalR.EoverP, hcalR.AvgHitEnergy,
                hcalR.SpreadPhi, hcalR.SpreadEta, hcalR.SpreadR, hcalR.MaxHitFrac,
                hcalR.EnergyStdDev, hcalR.EnergyConcentration,
                hcalR.R_Disp, hcalR.R_DispWeighted, hcalR.Eta_DispWeighted, hcalR.Phi_DispWeighted,

                tofF.Beta, tofF.MassSq, tofF.NHitsBarrel, tofF.NHitsEndcap, tofF.NHitsTotal,
                tofF.MinDistBarrel, tofF.MinDistEndcap, tofF.AvgLenBarrel, tofF.AvgLenEndcap, tofF.HasToF,

                static_cast<float>(trackP),
                static_cast<float>(trackEta),
                static_cast<float>(trackPhi)
            );


        // ---------------------------------------------------------------------
        // LowP ONNX inference
        //
        // Initialized once, not once per particle.
        // ---------------------------------------------------------------------

        static Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "MuonID"
        );

        static Ort::SessionOptions options;

        static std::unique_ptr<Ort::Session> session =
            []()
            {
                Ort::SessionOptions opts;

                opts.SetIntraOpNumThreads(1);

                opts.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::
                        ORT_ENABLE_EXTENDED
                );

                return std::make_unique<Ort::Session>(
                    env,
                    LOWP_ONNX_PATH,
                    opts
                );
            }();


        static Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );


        return run_muon_id(
            *session,
            memoryInfo,
            raw
        );
    }


    // =========================================================================
    // HIGH-P
    // =========================================================================

    else
    {
        // ---------------------------------------------------------------------
        // ECal
        // ---------------------------------------------------------------------

        const HighP_TrackCalHits ecalHits =
            HighP_CollectHits(
                simPart,
                ecalDetectors,
                *event,
                particle,
                TIMING_CUT_NS
            );

        const HighP_TrackFeatures ecalF =
            HighP_ComputeTrackFeatures(
                ecalHits,
                trackP
            );


        // ---------------------------------------------------------------------
        // HCal
        // ---------------------------------------------------------------------

        const HighP_TrackCalHits hcalHits =
            HighP_CollectHits(
                simPart,
                hcalDetectors,
                *event,
                particle,
                TIMING_CUT_NS
            );

        const HighP_TrackFeatures hcalF =
            HighP_ComputeTrackFeatures(
                hcalHits,
                trackP
            );


        // ---------------------------------------------------------------------
        // No calorimeter information -> not "found" for ID purposes.
        // ---------------------------------------------------------------------

        if (ecalF.Number <= 0.0f &&
            hcalF.Number <= 0.0f)
        {
            return 0.0f;
        }

        if (hasDetectorInfo != nullptr)
            *hasDetectorInfo = true;


        // ---------------------------------------------------------------------
        // 29-feature HighP vector
        // ---------------------------------------------------------------------

        const std::vector<float> raw =
            HighP_build_raw_features(
                ecalF.Energy,
                hcalF.Energy,

                ecalF.Number,
                hcalF.Number,

                ecalF.EoverP,
                hcalF.EoverP,

                ecalF.AvgHitEnergy,
                hcalF.AvgHitEnergy,

                ecalF.SpreadPhi,
                ecalF.SpreadEta,
                ecalF.SpreadR,

                hcalF.SpreadPhi,
                hcalF.SpreadEta,
                hcalF.SpreadR,

                ecalF.MaxHitFrac,
                hcalF.MaxHitFrac,

                ecalF.EnergyStdDev,
                hcalF.EnergyStdDev,

                ecalF.EnergyConcentration,
                hcalF.EnergyConcentration,

                ecalF.R_Disp,
                ecalF.R_DispWeighted,
                ecalF.Eta_DispWeighted,
                ecalF.Phi_DispWeighted,

                hcalF.R_Disp,
                hcalF.R_DispWeighted,
                hcalF.Eta_DispWeighted,
                hcalF.Phi_DispWeighted,

                static_cast<float>(trackP),
                static_cast<float>(trackEta)
            );


        // ---------------------------------------------------------------------
        // HighP ONNX inference
        // ---------------------------------------------------------------------

        static Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "MuonID"
        );

        static std::unique_ptr<Ort::Session> session =
            []()
            {
                Ort::SessionOptions opts;

                return std::make_unique<Ort::Session>(
                    env,
                    HIGHP_ONNX_PATH,
                    opts
                );
            }();


        static Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );


        return run_muon_id(
            *session,
            memoryInfo,
            raw
        );
    }
}