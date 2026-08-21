#include <TLorentzVector.h>
#include <TRandom.h>
static TRandom3 random_gen;
static double z_F = 174.0;  
static double rL_F =9.5;
static double rU_F =60.5;

static double r_B = 64.6; 
static double zL_B = -112.5;
static double zU_B = 147.5;

static double c_light = 29.9792458;

static double r_min = 0.0, r_max = 998.0, r_step = 2.0;
static double z_min = -800.0, z_max = 798.0, z_step = 2.0;
struct FieldRow {
    double r_field, z_field, br, bz;
};

static std::vector<FieldRow> field_data;

// Definicja pochodnych dla siły Lorentza
struct State { double x, y, z, px, py, pz; };

State GetDerivative(const State& s, double charge, double p_tot) {
    double r_curr = std::sqrt(s.x*s.x + s.y*s.y);
    int ir = std::round((r_curr - r_min) / r_step);
    int iz = std::round((s.z - z_min) / z_step);
    int n_z = (int)((z_max - z_min) / z_step) + 1;
    int index = ir * n_z + iz;

    double Br = 0, Bz = 0;
    if (index >= 0 && index < (int)field_data.size()) {
        Br = field_data[index].br;
        Bz = field_data[index].bz;
    }

    double Bx = (r_curr > 0) ? Br * (s.x / r_curr) : 0;
    double By = (r_curr > 0) ? Br * (s.y / r_curr) : 0;

    // k = q * c * step / p (w Twoich jednostkach)
    double f = 0.00029979 * charge; 
    
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
    double tof_ns;
    double tof_smear2;
    double tof_smear10;
    double tof_smear40;
    double track_length;
};

ToFResults ToFFastSim(TLorentzVector particle, int charge,double mass)
{
    //source /opt/detector/epic-main/bin/thisepic.sh 
    particle.SetPxPyPzE(particle.Px(),particle.Py(),particle.Pz(),sqrt(particle.P()*particle.P()+mass*mass));
    if (field_data.empty())
    {

        std::string path = "/opt/software/linux-x86_64_v2/epic-git.2886a4cb73093bfeffef25562329f79cb2dab196_main-eytkziklo45ilfu2iohqelyggommu4eh/share/epic/fieldmaps/MARCO_v.7.6.2.2.11_1.7T_Magnetic_Field_Map_2024_05_02_rad_coords_cm_T.BMap.txt";
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "There's no file!" << std::endl;
            return {0.0,0.0};
        }

        std::vector<FieldRow> data;
        double r_field, z_field, br, bz;

        // 1. Wczytywanie danych (to moze potrwac chwile, plik jest duzy)
        while (file >> r_field >> z_field >> br >> bz) {
        field_data.push_back({r_field, z_field, br, bz});
        }
        file.close();
    }
  
    State s = {0.0, 0.0, 0.0, particle.Px(), particle.Py(), particle.Pz()};
    double track_length = 0.0;
    double step = 0.2; // Możesz zwiększyć do 0.2-0.5 przy RK4 dla szybkości
    bool hit = false;
    double p_tot=particle.P();
    double beta=particle.P()/particle.E();
    while (track_length < 500.0) { // max 5 metrów
        State k1 = GetDerivative(s, charge, p_tot);
        
        State s2 = {s.x + k1.x*step/2, s.y + k1.y*step/2, s.z + k1.z*step/2, s.px + k1.px*step/2, s.py + k1.py*step/2, s.pz + k1.pz*step/2};
        State k2 = GetDerivative(s2, charge, p_tot);
        
        State s3 = {s.x + k2.x*step/2, s.y + k2.y*step/2, s.z + k2.z*step/2, s.px + k2.px*step/2, s.py + k2.py*step/2, s.pz + k2.pz*step/2};
        State k3 = GetDerivative(s3, charge, p_tot);
        
        State s4 = {s.x + k3.x*step, s.y + k3.y*step, s.z + k3.z*step, s.px + k3.px*step, s.py + k3.py*step, s.pz + k3.pz*step};
        State k4 = GetDerivative(s4, charge, p_tot);

        s.x += (step/6.0) * (k1.x + 2*k2.x + 2*k3.x + k4.x);
        s.y += (step/6.0) * (k1.y + 2*k2.y + 2*k3.y + k4.y);
        s.z += (step/6.0) * (k1.z + 2*k2.z + 2*k3.z + k4.z);
        s.px += (step/6.0) * (k1.px + 2*k2.px + 2*k3.px + k4.px);
        s.py += (step/6.0) * (k1.py + 2*k2.py + 2*k3.py + k4.py);
        s.pz += (step/6.0) * (k1.pz + 2*k2.pz + 2*k3.pz + k4.pz);

        // KLUCZOWE: Renormalizacja pędu (p_tot musi być stały!)
        double p_curr = std::sqrt(s.px*s.px + s.py*s.py + s.pz*s.pz);
        s.px *= (p_tot / p_curr);
        s.py *= (p_tot / p_curr);
        s.pz *= (p_tot / p_curr);

        track_length += step;
        double r_curr = std::sqrt(s.x*s.x + s.y*s.y);

        // Barrel ToF
        if (r_curr >= r_B && s.z < zU_B && s.z > zL_B) {
            hit = true;
            break;
        }
        // Forward ToF
        if (s.z >= z_F && r_curr > rL_F && r_curr < rU_F) {
            hit = true;
            break;
        }
        // Wyjście poza zakres
        if (r_curr > r_max || std::abs(s.z) > 800) break;
    }

    if (hit) {
        double tof_ns = track_length / (beta * c_light);
        //cout<<track_length<<endl;
        double tof_smear2=random_gen.Gaus(tof_ns,0.002);
        double tof_smear10=random_gen.Gaus(tof_ns,0.01);
        double tof_smear40=random_gen.Gaus(tof_ns,0.04);
        //return {tof_ns,track_length};
        return {tof_ns,tof_smear2,tof_smear10,tof_smear40,track_length};

    } 
    return {0.0,0.0,0.0,0.0,0.0};
}
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
double LeptonMass(double L, double p, double t)
{
    L=L/1.97327e-14;
    t=t/ 6.5821e-16;
    return p*p*((t*t)/(L*L)-1);
}