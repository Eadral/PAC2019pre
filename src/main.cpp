#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "DataStruct_Array.h"
#include <omp.h>
#include <time.h>
#include <pthread.h>
#include <thread>
#include <mkl.h>
#include <complex>
#include <tbb/tbb.h>
#include <cassert>


using namespace std;
using namespace FYSPACE;
using namespace tbb;

#ifndef _WIN32
const int ONE_D   = 1;
const int TWO_D   = 2;
const int THREE_D = 3;
const int ni      = 500;
const int nj      = 400;
const int nk      = 300;

#define F 2.2E3
#define Time 1E6
#else
const int scale = 10;
const int ONE_D = 1;
const int TWO_D = 2;
const int THREE_D = 3;
const int ni = 500 / scale;
const int nj = 400 / scale;
const int nk = 300;

#define F 1
#define Time 1
#endif



typedef double RDouble;
typedef FYArray<RDouble ,3> RDouble3D;
typedef FYArray<RDouble ,4> RDouble4D;

int preccheck(RDouble4D dqdx_4d,RDouble4D dqdy_4d,RDouble4D dqdz_4d);
int preccheck_small(RDouble4D dqdx_4d,RDouble4D dqdy_4d,RDouble4D dqdz_4d);

inline unsigned long long rdtsc(void)
{
	unsigned long hi = 0, lo = 0;

#ifndef _WIN32
	__asm__ __volatile__ ("lfence;rdtsc" : "=a"(lo), "=d"(hi));

	return (((unsigned long long)lo))|(((unsigned long long)hi)<<32);
#else

	return (unsigned long long)time(nullptr);;
#endif
}

inline void DoWork(RDouble4D dqdx_4d, int i_start, int i_end, int i_length, int j_start, int j_end, int j_length, int k_start, int k_end, int k_length, int m_start, int m_end, int m_length) {
	double *data = dqdx_4d.data();
	for (int m = m_start; m < m_end; m++) {
		double *data_m = data + m * k_length * j_length * i_length;
# pragma omp parallel for
		for (int k = k_start; k < k_end; k++) {
			double *data_k = data_m + k * j_length * i_length;
			for (int j = j_start; j < j_end; j++) {
				double *data_j = data_k + j * i_length;
				data_j += i_start;
#pragma omp simd
#pragma unroll
				for (int i = i_start; i < i_end; i++) {
					// int corr = dqdx_4d.getindex(i, j, k, m);
					// int now = data_j - dqdx_4d.data();
					*data_j++ = 0;
				}
			}
		}
	}
}

void DoWork13(const double fourth, RDouble4D xfn, RDouble4D yfn, RDouble4D zfn, RDouble4D area, RDouble4D q_4d, RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d, int i_start, int i_end, int i_length, int j_start, int j_end, int j_length, int k_start, int k_end,
	int k_length, int m_start, int m_end, int ns2, int il1, int il2, int jl1, int jl2, int kl1, int kl2) {
	const int i_unit = 256 + 128;
	const int j_unit = 2;
	const int k_unit = 2;
# pragma omp parallel for
	for (int k_div = k_start; k_div < k_end; k_div += k_unit) {
		int k_div_p = k_div + k_unit < k_end ? k_div + k_unit : k_end;
# pragma omp parallel for
		for (int j_div = j_start; j_div < j_end; j_div += j_unit) {
			int j_div_p = j_div + j_unit < j_end ? j_div + j_unit : j_end;

			for (int i_div = i_start; i_div < i_end; i_div += i_unit) {
				int i_div_p = i_div + i_unit < i_end ? i_div + i_unit : i_end;

				double* xfn_d = xfn.data();
				double* yfn_d = yfn.data();
				double* zfn_d = zfn.data();
				double* area_d = area.data();

				double* q_4d_d = q_4d.data();

				double* dqdx_4d_d = dqdx_4d.data();
				double* dqdy_4d_d = dqdy_4d.data();
				double* dqdz_4d_d = dqdz_4d.data();
				for (int m = m_start; m < m_end; m++) {
		
					double* xfn_m = xfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* yfn_m = yfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* zfn_m = zfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* xfn1_m = xfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* yfn1_m = yfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* zfn1_m = zfn_d + (ns2 - 1) * k_length * j_length * i_length;
					double* area_m = area_d + (ns2 - 1) * k_length * j_length * i_length;
					double* area1_m = area_d + (ns2 - 1) * k_length * j_length * i_length;

					double* q_4d_m = q_4d_d + m * k_length * j_length * i_length;
					double* dqdx_4d_m = dqdx_4d_d + m * k_length * j_length * i_length;
					double* dqdy_4d_m = dqdy_4d_d + m * k_length * j_length * i_length;
					double* dqdz_4d_m = dqdz_4d_d + m * k_length * j_length * i_length;
					double* dqdx_4d1_m = dqdx_4d_d + m * k_length * j_length * i_length;
					double* dqdy_4d1_m = dqdy_4d_d + m * k_length * j_length * i_length;
					double* dqdz_4d1_m = dqdz_4d_d + m * k_length * j_length * i_length;


					//# pragma omp parallel for
					for (int k = k_div; k < k_div_p; k++) {
	
						double* xfn_k = xfn_m + k * j_length * i_length;
						double* yfn_k = yfn_m + k * j_length * i_length;
						double* zfn_k = zfn_m + k * j_length * i_length;
						double* xfn1_k = xfn1_m + (k - kl1) * j_length * i_length;
						double* yfn1_k = yfn1_m + (k - kl1) * j_length * i_length;
						double* zfn1_k = zfn1_m + (k - kl1) * j_length * i_length;
						double* area_k = area_m + k * j_length * i_length;
						double* area1_k = area1_m + (k - kl1) * j_length * i_length;

						double* q_4d_k = q_4d_m + k * j_length * i_length;
						double* q_4d1_k = q_4d_m + (k - kl1) * j_length * i_length;
						double* q_4d2_k = q_4d_m + (k - kl2) * j_length * i_length;
						double* q_4d3_k = q_4d_m + (k - kl1 - kl2) * j_length * i_length;

						double* dqdx_4d_k = dqdx_4d_m + k * j_length * i_length;
						double* dqdy_4d_k = dqdy_4d_m + k * j_length * i_length;
						double* dqdz_4d_k = dqdz_4d_m + k * j_length * i_length;
						double* dqdx_4d1_k = dqdx_4d1_m + (k - kl2) * j_length * i_length;
						double* dqdy_4d1_k = dqdy_4d1_m + (k - kl2) * j_length * i_length;
						double* dqdz_4d1_k = dqdz_4d1_m + (k - kl2) * j_length * i_length;
						for (int j = j_div; j < j_div_p; j++) {
							double* xfn_j = xfn_k + j * i_length;
							double* yfn_j = yfn_k + j * i_length;
							double* zfn_j = zfn_k + j * i_length;
							double* xfn1_j = xfn1_k + (j - jl1) * i_length;
							double* yfn1_j = yfn1_k + (j - jl1) * i_length;
							double* zfn1_j = zfn1_k + (j - jl1) * i_length;
							double* area_j = area_k + j * i_length;
							double* area1_j = area1_k + (j - jl1) * i_length;

							xfn_j += i_div;
							yfn_j += i_div;
							zfn_j += i_div;
							xfn1_j += i_div - il1;
							yfn1_j += i_div - il1;
							zfn1_j += i_div - il1;
							area_j += i_div;
							area1_j += i_div - il1;

							double* q_4d_j = q_4d_k + j * i_length;
							double* q_4d1_j = q_4d1_k + (j - jl1) * i_length;
							double* q_4d2_j = q_4d2_k + (j - jl2) * i_length;
							double* q_4d3_j = q_4d3_k + (j - jl1 - jl2) * i_length;

							q_4d_j += i_div;
							q_4d1_j += i_div - il1;
							q_4d2_j += i_div - il2;
							q_4d3_j += i_div - il1 - il2;

							double* dqdx_4d_j = dqdx_4d_k + j * i_length;
							double* dqdy_4d_j = dqdy_4d_k + j * i_length;
							double* dqdz_4d_j = dqdz_4d_k + j * i_length;
							double* dqdx_4d1_j = dqdx_4d1_k + (j - jl2) * i_length;
							double* dqdy_4d1_j = dqdy_4d1_k + (j - jl2) * i_length;
							double* dqdz_4d1_j = dqdz_4d1_k + (j - jl2) * i_length;
							dqdx_4d_j += i_div;
							dqdy_4d_j += i_div;
							dqdz_4d_j += i_div;
							dqdx_4d1_j += i_div - il2;
							dqdy_4d1_j += i_div - il2;
							dqdz_4d1_j += i_div - il2;
#pragma omp simd
#pragma unroll
							for (int i = i_div; i < i_div_p; i++) {
								// int corr_worksx = worksx.getindex(i, j, k, 0);
								// int now_worksx = worksx_j - worksx.data();
								// int corr_xfn = xfn.getindex(i, j, k, ns2 - 1);
								// int now_xfn = xfn_j - xfn.data();
								// int corr_area = area.getindex(i, j, k, ns2 - 1);
								// int now_area = area_j - area.data();
								double ta = (*xfn_j++) * (*area_j) + (*xfn1_j++) * (*area1_j);
								double tb = (*yfn_j++) * (*area_j) + (*yfn1_j++) * (*area1_j);
								double tc = (*zfn_j++) * (*area_j) + (*zfn1_j++) * (*area1_j);
								area_j++;
								area1_j++;

								double t = fourth * (
									(*q_4d_j++) + (*q_4d1_j++) + (*q_4d2_j++) + (*q_4d3_j++)
								);

								double tx = ta * t;
								double ty = tb * t;
								double tz = tc * t;
								*dqdx_4d_j++ -= tx;
								*dqdy_4d_j++ -= ty;
								*dqdz_4d_j++ -= tz;
								*dqdx_4d1_j++ += tx;
								*dqdy_4d1_j++ += ty;
								*dqdz_4d1_j++ += tz;
							}
						}
					}
				}
			}
		}
	}
}

void DoWork14(RDouble3D vol, RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d, int i_start, int i_end, int i_length, int j_start, int j_end, int j_length, int k_start, int k_end, int k_length, int m_start, int m_end, int il1, int jl1, int kl1) {
	const int i_unit = 256;
	const int j_unit = 2;
	const int k_unit = 2;
# pragma omp parallel for
	for (int k_div = k_start; k_div < k_end; k_div += k_unit) {
		int k_div_p = k_div + k_unit < k_end ? k_div + k_unit : k_end;

		for (int j_div = j_start; j_div < j_end; j_div += j_unit) {
			int j_div_p = j_div + j_unit < j_end ? j_div + j_unit : j_end;

			for (int i_div = i_start; i_div < i_end; i_div += i_unit) {
				int i_div_p = i_div + i_unit < i_end ? i_div + i_unit : i_end;
			
				double *vol_d = vol.data();
				double *vol1_d = vol.data();

				double *dqdx_4d_d = dqdx_4d.data();
				double *dqdy_4d_d = dqdy_4d.data();
				double *dqdz_4d_d = dqdz_4d.data();
				for (int m = m_start; m < m_end; m++) {
					double *vol_m = vol_d;
					double *vol1_m = vol1_d;
			
					// # pragma omp parallel for

					double *dqdx_4d_m = dqdx_4d_d + m * k_length * j_length * i_length;
					double *dqdy_4d_m = dqdy_4d_d + m * k_length * j_length * i_length;
					double *dqdz_4d_m = dqdz_4d_d + m * k_length * j_length * i_length;
					for (int k = k_div; k < k_div_p; k++) {
						double *vol_k = vol_m + k * j_length * i_length;
						double *vol1_k = vol1_m + (k - kl1) * j_length * i_length;
					

						double *dqdx_4d_k = dqdx_4d_m + k * j_length * i_length;
						double *dqdy_4d_k = dqdy_4d_m + k * j_length * i_length;
						double *dqdz_4d_k = dqdz_4d_m + k * j_length * i_length;
						for (int j = j_div; j < j_div_p; j++) {
							double *vol_j = vol_k + j * i_length;
							double *vol1_j = vol1_k + (j - jl1) * i_length;
				
							vol_j += i_div;
							vol1_j += i_div - il1;
						

							double *dqdx_4d_j = dqdx_4d_k + j * i_length;
							double *dqdy_4d_j = dqdy_4d_k + j * i_length;
							double *dqdz_4d_j = dqdz_4d_k + j * i_length;
							dqdx_4d_j += i_div;
							dqdy_4d_j += i_div;
							dqdz_4d_j += i_div;
#pragma omp simd
#pragma unroll
							for (int i = i_div; i < i_div_p; i++) {
								// int corr = dqdx_4d.getindex(i, j, k, m);
								// int now = data_j - dqdx_4d.data();

								double t = 1.0 / (*vol_j++ + *vol1_j++);

								*dqdx_4d_j++ *= t;
								*dqdy_4d_j++ *= t;
								*dqdz_4d_j++ *= t;
							}
							// }
						}
					}
				}
			}
		}
	}
}

void DoWork15(RDouble4D xfn, RDouble4D yfn, RDouble4D zfn, RDouble4D area, RDouble4D q_4d, RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d, int i_start, int i_end, int i_length, int j_start, int j_end, int j_length, int k_start, int k_end, int k_length, int m_start,
	int m_end, int ns1, int il1, int jl1, int kl1) {
	const int i_unit = 256 + 128;
	const int j_unit = 2;
	const int k_unit = 2;
# pragma omp parallel for
	for (int k_div = k_start; k_div < k_end; k_div += k_unit) {
		int k_div_p = k_div + k_unit < k_end ? k_div + k_unit : k_end;
# pragma omp parallel for
		for (int j_div = j_start; j_div < j_end; j_div += j_unit) {
			int j_div_p = j_div + j_unit < j_end ? j_div + j_unit : j_end;

			for (int i_div = i_start; i_div < i_end; i_div += i_unit) {
				int i_div_p = i_div + i_unit < i_end ? i_div + i_unit : i_end;

			
				double* xfn_d = xfn.data();
				double* yfn_d = yfn.data();
				double* zfn_d = zfn.data();
				double* area_d = area.data();

				double* dqdx_4d_d = dqdx_4d.data();
				double* dqdy_4d_d = dqdy_4d.data();
				double* dqdz_4d_d = dqdz_4d.data();
				double* q_4d_d = q_4d.data();

				for (int m = m_start; m < m_end; m++) {

					double* xfn_m = xfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* yfn_m = yfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* zfn_m = zfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* xfn1_m = xfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* yfn1_m = yfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* zfn1_m = zfn_d + (ns1 - 1) * k_length * j_length * i_length;
					double* area_m = area_d + (ns1 - 1) * k_length * j_length * i_length;
					double* area1_m = area_d + (ns1 - 1) * k_length * j_length * i_length;
					//# pragma omp parallel for

					double* dqdx_4d_m = dqdx_4d_d + m * k_length * j_length * i_length;
					double* dqdy_4d_m = dqdy_4d_d + m * k_length * j_length * i_length;
					double* dqdz_4d_m = dqdz_4d_d + m * k_length * j_length * i_length;
					double* q_4d_m = q_4d_d + m * k_length * j_length * i_length;

					for (int k = k_div; k < k_div_p; k++) {
			
						double* xfn_k = xfn_m + k * j_length * i_length;
						double* yfn_k = yfn_m + k * j_length * i_length;
						double* zfn_k = zfn_m + k * j_length * i_length;
						double* xfn1_k = xfn1_m + (k - kl1) * j_length * i_length;
						double* yfn1_k = yfn1_m + (k - kl1) * j_length * i_length;
						double* zfn1_k = zfn1_m + (k - kl1) * j_length * i_length;
						double* area_k = area_m + k * j_length * i_length;
						double* area1_k = area1_m + (k - kl1) * j_length * i_length;

						double* dqdx_4d_k = dqdx_4d_m + k * j_length * i_length;
						double* dqdy_4d_k = dqdy_4d_m + k * j_length * i_length;
						double* dqdz_4d_k = dqdz_4d_m + k * j_length * i_length;
						double* dqdx_4d1_k = dqdx_4d_m + (k - kl1) * j_length * i_length;
						double* dqdy_4d1_k = dqdy_4d_m + (k - kl1) * j_length * i_length;
						double* dqdz_4d1_k = dqdz_4d_m + (k - kl1) * j_length * i_length;
						double* q_4d1_k = q_4d_m + (k - kl1) * j_length * i_length;
						for (int j = j_div; j < j_div_p; j++) {
				
							double* xfn_j = xfn_k + j * i_length;
							double* yfn_j = yfn_k + j * i_length;
							double* zfn_j = zfn_k + j * i_length;
							double* xfn1_j = xfn1_k + (j - jl1) * i_length;
							double* yfn1_j = yfn1_k + (j - jl1) * i_length;
							double* zfn1_j = zfn1_k + (j - jl1) * i_length;
							double* area_j = area_k + j * i_length;
							double* area1_j = area1_k + (j - jl1) * i_length;
				
							xfn_j += i_div;
							yfn_j += i_div;
							zfn_j += i_div;
							xfn1_j += i_div - il1;
							yfn1_j += i_div - il1;
							zfn1_j += i_div - il1;
							area_j += i_div;
							area1_j += i_div - il1;

							double* dqdx_4d_j = dqdx_4d_k + j * i_length;
							double* dqdy_4d_j = dqdy_4d_k + j * i_length;
							double* dqdz_4d_j = dqdz_4d_k + j * i_length;
							double* dqdx_4d1_j = dqdx_4d1_k + (j - jl1) * i_length;
							double* dqdy_4d1_j = dqdy_4d1_k + (j - jl1) * i_length;
							double* dqdz_4d1_j = dqdz_4d1_k + (j - jl1) * i_length;
							double* q_4d1_j = q_4d1_k + (j - jl1) * i_length;
							dqdx_4d_j += i_div;
							dqdy_4d_j += i_div;
							dqdz_4d_j += i_div;
							dqdx_4d1_j += i_div - il1;
							dqdy_4d1_j += i_div - il1;
							dqdz_4d1_j += i_div - il1;
							q_4d1_j += i_div - il1;
#pragma omp simd
#pragma unroll
							for (int i = i_div; i < i_div_p; i++) {
								// int corr_worksx = worksx.getindex(i, j, k, 0);
								// int now_worksx = worksx_j - worksx.data();
								// int corr_xfn = xfn.getindex(i, j, k, ns1 - 1);
								// int now_xfn = xfn_j - xfn.data();
								// int corr_area = area.getindex(i, j, k, ns1 - 1);
								// int now_area = area_j - area.data();
								double ta = (*xfn_j++) * (*area_j) + (*xfn1_j++) * (*area1_j);
								double tb = (*yfn_j++) * (*area_j) + (*yfn1_j++) * (*area1_j);
								double tc = (*zfn_j++) * (*area_j) + (*zfn1_j++) * (*area1_j);
								area_j++;
								area1_j++;

								double tx = ta * (*q_4d1_j);
								double ty = tb * (*q_4d1_j);
								double tz = tc * (*q_4d1_j);
								*dqdx_4d_j++ -= tx;
								*dqdy_4d_j++ -= ty;
								*dqdz_4d_j++ -= tz;
								*dqdx_4d1_j++ += tx;
								*dqdy_4d1_j++ += ty;
								*dqdz_4d1_j++ += tz;
								q_4d1_j++;
							}
						}
					}
				}
			}
		}
	}
}

int main()
{
	double start,end,elapsed;
	const int nDim = THREE_D;
	const double fourth = 0.25;
	int mst = 0;
	int med = 3;


	Range I(-1,ni+1);
	Range J(-1,nj+1);
	Range K(-1,nk+1);
	RDouble3D x(I, J, K, fortranArray);
	RDouble3D y(I, J, K, fortranArray);
	RDouble3D z(I, J, K, fortranArray);
	for ( int k = -1; k <= nk+1; ++ k )
	{
		for ( int j = -1; j <= nj+1; ++ j )
		{
			for ( int i = -1; i <= ni+1; ++ i )
			{
				x(i,j,k) = i*0.1;
				y(i,j,k) = j*0.2;
				z(i,j,k) = k*0.3;
			}
		}
	}


	// 申请变量空间
	I = Range(-1,ni+1);
	J = Range(-1,nj+1);
    K = Range(-1,nk+1);
    Range D(1,3);
	RDouble4D xfn (I,J,K,D,fortranArray);  // 网格单元↙左下面法向，D为方向
	RDouble4D yfn (I,J,K,D,fortranArray);
	RDouble4D zfn (I,J,K,D,fortranArray);
	RDouble4D area(I,J,K,D,fortranArray);  // 网格单元↙左下面面积
	RDouble3D vol (I,J,K,  fortranArray);  // 网格单元体积
	// RDouble3D vol_ (I,J,K,  fortranArray);  // 网格单元体积

    Range M(0,3); // 4个变量：速度u、v、w，温度T
    RDouble4D q_4d(I,J,K,M,fortranArray); // 存储流场量，位置在单元中心
	RDouble4D dqdx_4d(I,J,K,M,fortranArray); // 存储流场量计算得到的梯度偏x
	RDouble4D dqdy_4d(I,J,K,M,fortranArray); // 存储流场量计算得到的梯度偏y
	RDouble4D dqdz_4d(I,J,K,M,fortranArray); // 存储流场量计算得到的梯度偏z

	// 计算网格单元几何数据 xfn、fn、zfn、area、vol
	// 速度u、v、w，温度T 流场变量赋值，存储在q_4d中，便于后面速度、温度界面梯度计算
	// 程序每执行一个迭代步，流场变量被更新。此处给初场值u=1.0，v=0.0，w=0.0，T=1.0
	for ( int k = -1; k <= nk+1; ++ k )
	{
		for ( int j = -1; j <= nj+1; ++ j )
		{
			for ( int i = -1; i <= ni+1; ++ i )
			{
				xfn(i,j,k,1) = 1.0;
				xfn(i,j,k,2) = 0.0;
				xfn(i,j,k,3) = 0.0;
				yfn(i,j,k,1) = 0.0;
				yfn(i,j,k,2) = 1.0;
				yfn(i,j,k,3) = 0.0;
				zfn(i,j,k,1) = 0.0;
				zfn(i,j,k,2) = 0.0;
				zfn(i,j,k,3) = 1.0;
				area(i,j,k,1) = 0.06;
				area(i,j,k,2) = 0.03;
				area(i,j,k,3) = 0.02;
				vol(i,j,k) = 0.006;
			}
		}
	}
	for ( int k = -1; k <= nk+1; ++ k )
	{
		for ( int j = -1; j <= nj+1; ++ j )
		{
			for ( int i = -1; i <= ni+1; ++ i )
			{
				q_4d(i,j,k,0) = (x(i,j,k) * x(i,j,k) + y(i,j,k)*y(i,j,k)- 1.3164) / 2.1547; // u = a*x*x+b*y*y
				q_4d(i,j,k,1) = (z(i,j,k)*z(i,j,k) - 0.2157 ) * 0.137; // v=c*z*z
				q_4d(i,j,k,2) = (2.0*x(i,j,k) +  1.737) / 3.14; // w=d*x
				q_4d(i,j,k,3) = x(i,j,k) + y(i,j,k) + 1.3765; // T = x + y
			}
		}
	}
	start=rdtsc();
	//以上为数据初始化部分，不可修改！
	// --------------------------------------------------------------------
	// 求解速度、温度在“单元界面”上的梯度，i、j、k三个方向依次求解
	// 在程序中是“耗时部分”，每一个迭代步都会求解，以下为未优化代码
	// 希望参赛队伍在理解该算法的基础上，实现更高效的界面梯度求解，提升程序执行效率
	// --------------------------------------------------------------------
	// 此处开始统计计算部分代码运行时间



	Range IW(-1,ni+1);
	Range JW(-1,nj+1);
	Range KW(-1,nk+1);

	I = Range(1, ni + 1);
	J = Range(1, nj + 1);
	K = Range(1, nk + 1);

	int i_start = I.first() - IW.first();  int i_end = I.last() - IW.first() + 1; int i_length = I.length() - IW.first() + 1;
	int j_start = J.first() - JW.first();  int j_end = J.last() - JW.first() + 1; int j_length = J.length() - JW.first() + 1;
	int k_start = I.first() - KW.first();  int k_end = K.last() - KW.first() + 1; int k_length = K.length() - KW.first() + 1;
	int m_start = M.first();  int m_end = M.last() + 1; int m_length = M.length();

	for (int nsurf = 1; nsurf <= THREE_D; ++ nsurf )	{

		int index[] = {1,2,3,1,2};

		int ns1 = nsurf;
		int ns2 = index[nsurf  ];
		int ns3 = index[nsurf+1];

		int il1 = 0;
		int il2 = 0;
		int il3 = 0;
		int jl1 = 0;
		int jl2 = 0;
		int jl3 = 0;
		int kl1 = 0;
		int kl2 = 0;
		int kl3 = 0;

		if ( nsurf == 1 ) {
			il1 = 1;
			jl2 = 1;
			kl3 = 1;
		} else if ( nsurf == 2 ) {
			jl1 = 1;
			kl2 = 1;
			il3 = 1;
		} else if ( nsurf == 3 ) {
			kl1 = 1;
			il2 = 1;
			jl3 = 1;
		}

		DoWork(dqdx_4d, i_start, i_end, i_length, j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end,
		       m_length);
		DoWork(dqdy_4d, i_start, i_end, i_length, j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end,
		       m_length);
		DoWork(dqdz_4d, i_start, i_end, i_length, j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end,
		       m_length);

		DoWork15(xfn, yfn, zfn, area, q_4d, dqdx_4d, dqdy_4d, dqdz_4d, i_start, i_end, i_length,
		         j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end, ns1, il1, jl1, kl1);


		if ((nsurf != 2) || (nDim != TWO_D)) {
			DoWork13(fourth, xfn, yfn, zfn, area, q_4d, dqdx_4d, dqdy_4d, dqdz_4d,
			         i_start, i_end,
			         i_length, j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end, ns2, il1, il2, jl1,
			         jl2, kl1,
			         kl2);
		}

		if ((nsurf != 1) || (nDim != TWO_D)) {
			DoWork13(fourth, xfn, yfn, zfn, area, q_4d, dqdx_4d, dqdy_4d, dqdz_4d,
			         i_start, i_end,
			         i_length, j_start, j_end, j_length, k_start, k_end, k_length, m_start, m_end, ns3, il1, il3, jl1,
			         jl3, kl1,
			         kl3);
		}

		DoWork14(vol, dqdx_4d, dqdy_4d, dqdz_4d, i_start, i_end, i_length, j_start, j_end, j_length, k_start,
		         k_end,
		         k_length, m_start, m_end, il1, jl1, kl1);
		
	// 该方向界面梯度值被计算出来后，会用于粘性通量计算，该值使用后下一方向会重新赋0计算

	}

	//----------------------------------------------------
	//以下为正确性对比部分，不可修改！
	//----------------------------------------------------
	end=rdtsc();
	elapsed= (end - start)/(F*Time);
	cout<<"The programe elapsed "<<elapsed<<setprecision(8)<<" s"<<endl;
#ifdef _WIN32
	if (!preccheck_small(dqdx_4d, dqdy_4d, dqdz_4d))
		cout << "Result check passed!" << endl;
	return 0;

#else

	if(!preccheck(dqdx_4d,dqdy_4d,dqdz_4d))
		cout<<"Result check passed!"<<endl;
	return 0;
#endif
}

int preccheck(RDouble4D dqdx_4d,RDouble4D dqdy_4d,RDouble4D dqdz_4d)
{
	double tmp,real;
	ifstream file("check.txt",std::ofstream::binary);
	if ( !file )
	{
		cout << "Error opening check file! ";
		exit(1);
	}
    	for ( int i = 0; i < ni; ++ i )
	{
    		for ( int j = 0; j < nj; ++ j )
		{
			for ( int k = 0; k < nk; ++ k )
    			{
				for (int m = 0; m < 3; ++ m)
    				{
					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if(fabs(dqdx_4d(i,j,k,m) - tmp) > 1e-6)
					{
						real = dqdx_4d(i,j,k,m);
						cout<<"Precision check failed !"<<endl;
						cout<<"Your result is "<<setprecision(15)<<real<<endl;
						cout<<"The Standard result is "<<setprecision(15)<<tmp<<endl;
						cout<<"The wrong position is "<<endl;
						cout<<"i="<<i<<",j="<<j<<",k="<<k<<",m="<<m<<endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if(fabs(dqdy_4d(i,j,k,m) - tmp) > 1e-6)
					{
						real = dqdy_4d(i,j,k,m);
						cout<<"Precision check failed !"<<endl;
						cout<<"Your result is "<<setprecision(15)<<real<<endl;
						cout<<"The Standard result is "<<setprecision(15)<<tmp<<endl;
						cout<<"The wrong position is "<<endl;
						cout<<"i="<<i<<",j="<<j<<",k="<<k<<",m="<<m<<endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if(fabs(dqdz_4d(i,j,k,m) - tmp) >1e-6)
					{
						real = dqdz_4d(i,j,k,m);
						cout<<"Precision check failed !"<<endl;
						cout<<"Your result is "<<setprecision(15)<<real<<endl;
						cout<<"The Standard result is "<<setprecision(15)<<tmp<<endl;
						cout<<"The wrong position is "<<endl;
						cout<<"i="<<i<<",j="<<j<<",k="<<k<<",m="<<m<<endl;
						exit(1);
					}
				}
			}
		}
	}
	file.close();
	return 0;
}


int preccheck_small(RDouble4D dqdx_4d, RDouble4D dqdy_4d, RDouble4D dqdz_4d)
{
	double tmp, real;
	ifstream file("check_small.txt", std::ofstream::binary);
	if (!file)
	{
		cout << "Error opening check file! ";
		exit(1);
	}
	for (int i = 0; i < ni; ++i)
	{
		for (int j = 0; j < nj; ++j)
		{
			for (int k = 0; k < nk; ++k)
			{
				for (int m = 0; m < 3; ++m)
				{
					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdx_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdx_4d(i, j, k, m);
						cout << "Precision check failed x!" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdy_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdy_4d(i, j, k, m);
						cout << "Precision check failed y!" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}

					file.read(reinterpret_cast<char*>(&tmp), sizeof(double));
					if (fabs(dqdz_4d(i, j, k, m) - tmp) > 1e-6)
					{
						real = dqdz_4d(i, j, k, m);
						cout << "Precision check failed z!" << endl;
						cout << "Your result is " << setprecision(15) << real << endl;
						cout << "The Standard result is " << setprecision(15) << tmp << endl;
						cout << "The wrong position is " << endl;
						cout << "i=" << i << ",j=" << j << ",k=" << k << ",m=" << m << endl;
						exit(1);
					}
				}
			}
		}
	}
	file.close();
	return 0;
}
