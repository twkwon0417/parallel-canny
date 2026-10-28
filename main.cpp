// main.cpp
// OpenMP Canny Edge Detector Assignment Skeleton
// Build: g++ -fopenmp -march=native main.cpp -o canny
// Usage: ./canny [입력 파일명] [출력 파일명] [쓰레드 수]
// Timing excludes I/O and verification. Verification is PASS/FAIL against a reference.

#include <fstream>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <omp.h>
#include <vector>

#include "student_kernel.hpp"

using namespace std;

// --------------------- Simple PGM I/O (P5, 8-bit) ---------------------
static void skip_comments(istream& in){
    int c;
    while((c=in.peek())=='#'){
        string dummy; getline(in,dummy);
    }
}
static vector<uint8_t> read_pgm(const string& path, int& W, int& H){
    ifstream in(path, ios::binary);
    if(!in) throw runtime_error("Failed to open input: " + path);
    string magic; in >> magic; 
    if(magic != "P5") throw runtime_error("Only P5 PGM supported");
    skip_comments(in);
    in >> W; skip_comments(in); in >> H; skip_comments(in);
    int maxv; in >> maxv;
    if(maxv != 255) throw runtime_error("Only 8-bit PGM supported");
    in.get(); // single whitespace
    vector<uint8_t> img((size_t)W*H);
    in.read(reinterpret_cast<char*>(img.data()), img.size());
    if(!in) throw runtime_error("PGM read error");
    return img;
}
static void write_pgm(const string& path, const vector<uint8_t>& img, int W, int H){
    ofstream out(path, ios::binary);
    if(!out) throw runtime_error("Failed to open output: " + path);
    out << "P5\n" << W << " " << H << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data()), img.size());
}

// --------------------- Utility ---------------------
template<typename T>
static inline T clampv(T x, T lo, T hi){ return x<lo?lo:(x>hi?hi:x); }

// --------------------- Reference (Single-threaded) ---------------------
namespace reference {

// 5x5 separable Gaussian kernel (sigma ~1.0)
static void gaussian5x1(const vector<uint8_t>& in, vector<float>& tmp, int W, int H){
    static const float k[5] = {1,4,6,4,1};
    const float norm = 1.0f/16.0f;
    // horizontal
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float s=0.0f;
            for(int dx=-2;dx<=2;++dx){
                int xx = clampv(x+dx,0,W-1);
                s += k[dx+2]*in[(size_t)y*W+xx];
            }
            tmp[(size_t)y*W+x] = s*norm;
        }
    }
    // vertical (in-place to tmp2)
}
static void gaussian1x5_from_tmp(const vector<float>& tmp, vector<float>& out, int W, int H){
    static const float k[5] = {1,4,6,4,1};
    const float norm = 1.0f/16.0f;
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float s=0.0f;
            for(int dy=-2;dy<=2;++dy){
                int yy = clampv(y+dy,0,H-1);
                s += k[dy+2]*tmp[(size_t)yy*W+x];
            }
            out[(size_t)y*W+x] = s*norm;
        }
    }
}

static void sobel(const vector<float>& in, vector<float>& gx, vector<float>& gy, int W, int H){
    const int dx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int dy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float sx=0, sy=0;
            for(int j=-1;j<=1;++j){
                int yy = clampv(y+j,0,H-1);
                for(int i=-1;i<=1;++i){
                    int xx = clampv(x+i,0,W-1);
                    float v = in[(size_t)yy*W+xx];
                    sx += dx[j+1][i+1]*v;
                    sy += dy[j+1][i+1]*v;
                }
            }
            gx[(size_t)y*W+x]=sx; gy[(size_t)y*W+x]=sy;
        }
    }
}

static void grad_mag_dir(const vector<float>& gx, const vector<float>& gy,
                         vector<float>& mag, vector<uint8_t>& dir, int W, int H){
    for(int i=0;i<W*H;++i){
        float sx=gx[i], sy=gy[i];
        mag[i] = std::hypot(sx,sy);
        // quantize to 0,45,90,135 based on atan2
        float a = atan2f(sy, sx) * 180.0f / float(M_PI);
        if(a<0) a+=180.0f;
        uint8_t q;
        if(a<22.5 || a>=157.5) q=0;
        else if(a<67.5) q=45;
        else if(a<112.5) q=90;
        else q=135;
        dir[i]=q;
    }
}

static void nonmax_supp(const vector<float>& mag, const vector<uint8_t>& dir,
                        vector<float>& thin, int W, int H){
    for(int y=0;y<H;++y){
        for(int x=0;x<W;++x){
            float m = mag[(size_t)y*W+x];
            float m1=0,m2=0;
            uint8_t d = dir[(size_t)y*W+x];
            if(d==0){ // left-right
                m1 = mag[(size_t)y*W+clampv(x-1,0,W-1)];
                m2 = mag[(size_t)y*W+clampv(x+1,0,W-1)];
            } else if(d==45){
                m1 = mag[(size_t)clampv(y-1,0,H-1)*W + clampv(x+1,0,W-1)];
                m2 = mag[(size_t)clampv(y+1,0,H-1)*W + clampv(x-1,0,W-1)];
            } else if(d==90){
                m1 = mag[(size_t)clampv(y-1,0,H-1)*W + x];
                m2 = mag[(size_t)clampv(y+1,0,H-1)*W + x];
            } else { // 135
                m1 = mag[(size_t)clampv(y-1,0,H-1)*W + clampv(x-1,0,W-1)];
                m2 = mag[(size_t)clampv(y+1,0,H-1)*W + clampv(x+1,0,W-1)];
            }
            thin[(size_t)y*W+x] = (m>=m1 && m>=m2) ? m : 0.0f;
        }
    }
}

static void double_threshold(const vector<float>& thin, vector<uint8_t>& edges,
                             float lowT, float highT, int W, int H){
    for(int i=0;i<W*H;++i){
        float v = thin[i];
        if(v >= highT) edges[i]=2; // strong
        else if(v >= lowT) edges[i]=1; // weak
        else edges[i]=0;
    }
}

static void hysteresis(vector<uint8_t>& edges, int W, int H){
    // BFS from strong edges; promote connected weak pixels
    deque<int> dq;
    auto idx = [W](int y,int x){return y*W+x;};
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        if(edges[idx(y,x)]==2) dq.push_back(idx(y,x));
    }
    const int dx[8]={-1,0,1,-1,1,-1,0,1};
    const int dy[8]={-1,-1,-1,0,0,1,1,1};
    while(!dq.empty()){
        int p=dq.front(); dq.pop_front();
        int y=p/W, x=p%W;
        for(int k=0;k<8;++k){
            int nx=x+dx[k], ny=y+dy[k];
            if(nx<0||ny<0||nx>=W||ny>=H) continue;
            int q = idx(ny,nx);
            if(edges[q]==1){ edges[q]=2; dq.push_back(q); }
        }
    }
    // finalize: 255 for edges, 0 otherwise
    for(int i=0;i<W*H;++i) edges[i] = (edges[i]==2)?255:0;
}

static void canny_reference(const vector<uint8_t>& in, vector<uint8_t>& out, int W, int H){
    vector<float> tmp(W*H), blur(W*H), gx(W*H), gy(W*H), mag(W*H), thin(W*H);
    vector<uint8_t> dir(W*H), edges(W*H);
    gaussian5x1(in, tmp, W, H);
    gaussian1x5_from_tmp(tmp, blur, W, H);
    sobel(blur, gx, gy, W, H);
    grad_mag_dir(gx, gy, mag, dir, W, H);
    nonmax_supp(mag, dir, thin, W, H);
    // auto thresholds using Otsu-like simple rule: hi=mean+std, lo=0.4*hi
    double sum=0, sum2=0;
    for(float v:thin){ sum+=v; sum2+=v*v; }
    double mean = sum/(W*H);
    double var = sum2/(W*H)-mean*mean;
    double stdv = std::sqrt(max(0.0, var));
    float highT = float(mean + stdv);
    float lowT = highT*0.4f;
    double_threshold(thin, edges, lowT, highT, W, H);
    hysteresis(edges, W, H);
    out = std::move(edges);
}

} // namespace reference

// --------------------- Verification ---------------------
static bool verify_pass(const vector<uint8_t>& ref, const vector<uint8_t>& test, int W, int H, double& mismatch_rate){
    size_t mismatch=0;
    for(size_t i=0;i<ref.size();++i){
        if(ref[i]!=test[i]) ++mismatch;
    }
    mismatch_rate = 100.0 * double(mismatch) / double(ref.size());
    // allow up to 1.0% mismatch due to tie-breaks; tweak if needed
    return mismatch_rate <= 1.0;
}

// --------------------- Main ---------------------
int main(int argc, char** argv){
    if(argc < 3){
        cerr << "Usage: " << argv[0] << " input.pgm output.pgm [threads]\n";
        return 1;
    }
    string in_path = argv[1], out_path = argv[2];
    int user_threads = (argc>=4) ? max(1, atoi(argv[3])) : omp_get_max_threads();

    int W=0,H=0;
    auto img = read_pgm(in_path, W, H);

    // 버퍼 준비
    vector<uint8_t> out_ref(W*H), out_student(W*H);

    // 단일 쓰레드(직렬 코드)의 실행 시간 측정
    const int N = 5;
    std::vector<double> ref_ms;
    ref_ms.reserve(N);
    for(int r=0;r<N;++r){
        double t0 = omp_get_wtime();
        reference::canny_reference(img, out_ref, W, H);
        double t1 = omp_get_wtime();
        ref_ms.push_back((t1-t0)*1000.0);
    }
    std::nth_element(ref_ms.begin(), ref_ms.begin()+N/2, ref_ms.end());
    double ref_med_ms = ref_ms[N/2];

    // 학생 구현 알고리즘의 실행시간만 계산
    omp_set_num_threads(user_threads);
    double t2 = omp_get_wtime();
    student::canny(img, out_student, W, H);
    double t3 = omp_get_wtime();

    // 검증 단계
    double mismatch_rate = 0.0;
    bool ok = verify_pass(out_ref, out_student, W, H, mismatch_rate);

    // 출력물 생성
    write_pgm(out_path, out_student, W, H);

    // 최종 결과 확인
    cout << "Input: " << in_path << " ("<<W<<"x"<<H<<")\n";
    cout << "Threads: " << user_threads << "\n";
    cout << "Reference time: " << ref_med_ms << " ms\n";
    cout << "Student time: " << (t3-t2)*1000 << " ms\n";
    cout << "PASS: " << (ok? "PASS\t":"FAIL\t");
    if(mismatch_rate >= 0.01) cout << "  (Mismatch " << mismatch_rate << " %)\n";

    return ok?0:2;
}
