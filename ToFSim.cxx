#include <TLorentzVector.h>
#include <TRandom.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>


static double c_light = 299.792458; // mm / ns  -- ALL LENGTHS IN THIS FILE ARE IN MM

double computeExpectedTime(double mass, double p, double L) {
    // L must be in mm (matches c_light units above)
    double beta = p / sqrt(p*p + mass*mass);
    return L / (beta * c_light);
}

struct FieldRow {
    double r_field, z_field, br, bz;
};

// Field map grid is stored in the file in CENTIMETERS ("rad_coords_cm_T").
// Track positions (s.x, s.y, s.z) are propagated in MILLIMETERS to match
// EDM4hep hit positions and computeExpectedTime(). Conversion happens only
// at the point of grid lookup (GetDerivative), nowhere else.
static double r_min = 0.0, r_max = 998.0, r_step = 2.0;   // cm
static double z_min = -800.0, z_max = 798.0, z_step = 2.0; // cm

static std::vector<FieldRow> field_data;

// Definicja pochodnych dla sily Lorentza
struct State { double x, y, z, px, py, pz; };

State GetDerivative(const State& s, double charge, double p_tot) {
    double r_curr_mm = std::sqrt(s.x*s.x + s.y*s.y);

    // Convert to cm to index the field grid (grid is in cm)
    double r_curr_cm = r_curr_mm / 10.0;
    double z_curr_cm = s.z / 10.0;

    int n_r = (int)((r_max - r_min) / r_step) + 1;
    int n_z = (int)((z_max - z_min) / z_step) + 1;

    int ir = (int)std::floor((r_curr_cm - r_min) / r_step + 0.5);
    int iz = (int)std::floor((z_curr_cm - z_min) / z_step + 0.5);

    double Br = 0, Bz = 0;
    if (ir >= 0 && ir < n_r && iz >= 0 && iz < n_z) {
        int index = ir * n_z + iz;
        if (index >= 0 && index < (int)field_data.size()) {
            Br = field_data[index].br;
            Bz = field_data[index].bz;
        }
    }
    // else: outside the map -> Br = Bz = 0 (field-free region), which is the
    // correct fallback rather than silently reading an unrelated grid cell.

    double Bx = (r_curr_mm > 0) ? Br * (s.x / r_curr_mm) : 0;
    double By = (r_curr_mm > 0) ? Br * (s.y / r_curr_mm) : 0;

    // k = q * c * step / p, calibrated for step/positions in MILLIMETERS.
    // (0.299792458 is the standard constant for meters; /1000 -> mm.)
    double f = 0.000299792458 * charge;

    return {
        s.px / p_tot,           // dx/ds
        s.py / p_tot,           // dy/ds
        s.pz / p_tot,           // dz/ds
        (s.py * Bz - s.pz * By) * f / p_tot, // dpx/ds
        (s.pz * Bx - s.px * Bz) * f / p_tot, // dpy/ds
        (s.px * By - s.py * Bx) * f / p_tot  // dpz/ds
    };
}

struct ToFResults {
    double track_length;
    bool DistanceCheck;
    double distance_to_TOF;
};

ToFResults ToFSim(TLorentzVector particle, int charge, TVector3 TOF_pos)
{
    //source /opt/detector/epic-main/bin/thisepic.sh

    TVector3 particle_mom = particle.Vect();
    if (field_data.empty())
    {
        std::string path = "/opt/software/linux-x86_64_v2/epic-git.6644324d24120168ad5e5eb725a77ea1d542b87f_main-djwkjya7qg4uxg375f4y6qwf52vld7sr/share/epic/fieldmaps/MARCO_v.7.6.2.2.11_1.7T_Magnetic_Field_Map_2024_05_02_rad_coords_cm_T.BMap.txt";
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "There's no file!" << std::endl;
            return {0.0, false, 0.0};
        }

        std::vector<FieldRow> data;
        double r_field, z_field, br, bz;

        // 1. Wczytywanie danych (to moze potrwac chwile, plik jest duzy)
        while (file >> r_field >> z_field >> br >> bz) {
            field_data.push_back({r_field, z_field, br, bz});
        }
        file.close();
    }

    State s = {0.0, 0.0, 0.0, particle_mom.Px(), particle_mom.Py(), particle_mom.Pz()};
    double track_length = 0.0;
    double best_length = 0.0;     // length at the point of closest approach
    double step = 0.2;            // mm
    bool DistanceCheck = false;
    double p_tot = particle_mom.Mag();
    double smallest_distance = std::numeric_limits<double>::max();

    while (track_length < 2100.0) { // max ~2.1 m (positions in mm)
        State k1 = GetDerivative(s, charge, p_tot);

        State s2 = {s.x + k1.x*step/2, s.y + k1.y*step/2, s.z + k1.z*step/2,
                    s.px + k1.px*step/2, s.py + k1.py*step/2, s.pz + k1.pz*step/2};
        State k2 = GetDerivative(s2, charge, p_tot);

        State s3 = {s.x + k2.x*step/2, s.y + k2.y*step/2, s.z + k2.z*step/2,
                    s.px + k2.px*step/2, s.py + k2.py*step/2, s.pz + k2.pz*step/2};
        State k3 = GetDerivative(s3, charge, p_tot);

        State s4 = {s.x + k3.x*step, s.y + k3.y*step, s.z + k3.z*step,
                    s.px + k3.px*step, s.py + k3.py*step, s.pz + k3.pz*step};
        State k4 = GetDerivative(s4, charge, p_tot);

        s.x += (step/6.0) * (k1.x + 2*k2.x + 2*k3.x + k4.x);
        s.y += (step/6.0) * (k1.y + 2*k2.y + 2*k3.y + k4.y);
        s.z += (step/6.0) * (k1.z + 2*k2.z + 2*k3.z + k4.z);
        s.px += (step/6.0) * (k1.px + 2*k2.px + 2*k3.px + k4.px);
        s.py += (step/6.0) * (k1.py + 2*k2.py + 2*k3.py + k4.py);
        s.pz += (step/6.0) * (k1.pz + 2*k2.pz + 2*k3.pz + k4.pz);

        // KLUCZOWE: Renormalizacja pedu (p_tot musi byc staly!)
        double p_curr = std::sqrt(s.px*s.px + s.py*s.py + s.pz*s.pz);
        s.px *= (p_tot / p_curr);
        s.py *= (p_tot / p_curr);
        s.pz *= (p_tot / p_curr);

        track_length += step;

        double distance_to_TOF = std::sqrt(
            (s.x - TOF_pos.X())*(s.x - TOF_pos.X()) +
            (s.y - TOF_pos.Y())*(s.y - TOF_pos.Y()) +
            (s.z - TOF_pos.Z())*(s.z - TOF_pos.Z()));

        if (distance_to_TOF < smallest_distance) {
            smallest_distance = distance_to_TOF;
            best_length = track_length;   // record length AT the best point
        } else {
            break;
        }
    }

    if (smallest_distance < 300.0) {
        DistanceCheck = true;
    }
    return {best_length, DistanceCheck, smallest_distance};
}
/*
double DiLeptonMass(double L1, double L2, double p1, double p2, double dt)
{
    L1=L1/1.97327e-14;
    L2=L2/1.97327e-14;
    dt=dt/ 6.5821e-16;

    double A=-2*(L1*L1)*(L2*L2)/((p1*p1)*(p2*p2))+pow(L1,4)/pow(p1,4)+pow(L2,4)/pow(p2,4);
    double B=-2*(L1*L1)*(L2*L2)*((1/(p1*p1))+(1/(p2*p2)))+2*pow(L1,4)/(p1*p1)+2*pow(L2,4)/(p2*p2)-2*(dt*dt)*((L1*L1)/(p1*p1)+(L2*L2)/(p2*p2));
    double C=pow(dt,4)-2*(dt*dt)*((L1*L1)+(L2*L2))+pow(L1,4)+pow(L2,4)-2*(L1*L1)*(L2*L2);
    return (-B+sqrt((B*B)-4*A*C))/(2*A);
}
*/