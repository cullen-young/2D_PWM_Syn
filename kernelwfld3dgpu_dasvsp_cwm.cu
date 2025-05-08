/* Kernels for 3D Acou/elastic wave propagation on GPUs */

/*
  Copyright (C) 2014 University of Western Australia

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#include "cudaErrors.cu"
#include <curand.h>
#include <curand_kernel.h>

// . . The basic block size
#define COEFFS_SIZE 5
#define RAD 4  // half of the order in space (k/2) 
#define DIAM 9
#define RADIUS 4

#define BDIMX 16
#define BDIMY 16
#define BDIMT 256

#define NH2 17 /* Maximum size of shifting (i.e. nh2=65) */

// . . Define 5s-element coefficient array and store in CONSTANT memory
__constant__ float cc2_d[COEFFS_SIZE];
__constant__ float cc1_d[COEFFS_SIZE];
__constant__ float cc12_d[3];
 
// . . Define functions constant to the program
// . . Constant memory is more optimal!
__constant__ int nx_d;
__constant__ int ny_d;
__constant__ int nz_d;
__constant__ int nr_d;
__constant__ int nh_d;
__constant__ int nt_d;
__constant__ int np_d;
__constant__ int nsx_d;
__constant__ int nsy_d;
__constant__ int wfld_j_d;
__constant__ float dx_d;
__constant__ float dy_d;
__constant__ float dz_d;
__constant__ float dt_d;
__constant__ float ox_d;
__constant__ float oy_d;
__constant__ float oz_d;
__constant__ float a_d;
__constant__ float eps_d;
__constant__ float cc_d[COEFFS_SIZE];

/*------------------------------------------------------------*/
/* . . Inject wavefield into a 3D volume     			      */
/*------------------------------------------------------------*/
__global__ void inject_3d(int iz,
			  float *inj,
			  float *wfld2,
			  int it,
			  int dat_j)
{
  // . . Coordinate indexing - Global coordinates
  int ix = blockIdx.x*blockDim.x+threadIdx.x; // x
  int iy = blockIdx.y*blockDim.y+threadIdx.y; // y
  int block=nx_d*ny_d;

  if ( ix < nx_d && iy < ny_d ) {
    int iaddr= iy * nx_d + ix;
    float w1 = (float)(it%dat_j) / (float)(dat_j);
    float w2 = abs(1.f-w1);
    wfld2[iz*block+iaddr] += w1*inj[iaddr] + w2*inj[block+iaddr];
  }
}

/*------------------------------------------------------------*/
/* . . Inject at a single point								  */
/*------------------------------------------------------------*/
__global__ void inject(	int iz,
			int ix,
			int iy,
                       	float *inj,
                    	float *wfld2,
                 		int it)
/*< Inject at a single point>*/
{

  if ( ix < nx_d && iy < ny_d ) {
    int iaddr= iz*nx_d*ny_d+iy*nx_d+ix;
    //int w1x = ((sou[3*is+0]-ox_d)/dx_d) - int((sou[3*is+0]-ox_d)/dx_d);
    //int w2x = 1 - w1x;
    //int w1y = ((sou[3*is+1]-oy_d)/dy_d) - int((sou[3*is+1]-oy_d)/dy_d);
    //int w2y = 1 - w1y;
    //int w1z = ((sou[3*is+2]-oz_d)/dz_d) - int((sou[3*is+2]-oz_d)/dz_d);
    //int w2z = 1 - w1z;
    wfld2[iaddr] += inj[it];
  }
}

/*------------------------------------------------------------*/
/* . . Inject on a grid                                       */
/*------------------------------------------------------------*/
__global__ void inject_grid(int it,
                                float *rec,
				float *inj,
				float *wfld)

{
  int ir = blockIdx.x*blockDim.x+threadIdx.x;
  
  int ix = (rec[3*ir+0]-ox_d)/dx_d;
  int iy = (rec[3*ir+1]-oy_d)/dy_d;
  int iz = (rec[3*ir+2]-oz_d)/dz_d;

  if (ir < nr_d) {
    int iaddr = iz*nx_d*ny_d + iy*nx_d + ix;
    wfld[iaddr] += inj[it*nr_d+ir];
  }
}
/*------------------------------------------------------------*/
/* . . Inject on a plane at 1 depth							  */
/*------------------------------------------------------------*/
__global__ void inject_plane(	int iz,
                       			float *inj,
                    			float *wfld2)
/*< Inject on a single plane>*/
{
  int ix = blockIdx.x*blockDim.x+threadIdx.x; // x
  int iy = blockIdx.y*blockDim.y+threadIdx.y; // y

  if ( ix < nx_d && iy < ny_d ) {
	int block = ny_d*nx_d;
    int ixy = iy*nx_d+ix;
    int iaddr= iz*block+ixy;
    wfld2[iaddr] += inj[ixy];
  }
}

/*------------------------------------------------------------*/
/* . . Inject on a plane at 1 depth                                                       */
/*------------------------------------------------------------*/
__global__ void inject_kernel_receiver_3d(int iz,
										  int it,
                                          float *inj,
                                          float *wfld2,
										  int dat_j)
/*< Inject wavefield on plane at depth iz>*/
{
  int ix = blockIdx.x*blockDim.x+threadIdx.x; // x
  int iy = blockIdx.y*blockDim.y+threadIdx.y; // y
  
  if ( ix < nx_d && iy < ny_d ) {
    float wt1 = (float)(it%dat_j) / (float)(dat_j);
    float wt2 = (1.f-wt1);
    int taddr1 =  it   *ny_d*nx_d+iy*nx_d+ix;
    int taddr2 = (it+1)*ny_d*nx_d+iy*nx_d+ix;
    int iaddr  =     iz*ny_d*nx_d+iy*nx_d+ix;
    wfld2[iaddr] += wt1*inj[taddr1]+wt2*inj[taddr2];
  }
  
}

/*------------------------------------------------------------*/
/* . . Inject on a vsp plane                                                       */
/*------------------------------------------------------------*/
__global__ void inject_linearborehole(   int ix, int iy,
                                        float *inj,
                                        float *wfld)
/*< Inject on a single plane>*/
{
  int iz = blockIdx.x*blockDim.x+threadIdx.x; // z

  if ( iz < nz_d) {
    int iaddr= iz*nx_d*ny_d+iy*nx_d+ix;
    wfld[iaddr] += inj[iaddr];
  }
}


/*------------------------------------------------------------*/
/* . . Inject on a vsp plane                                                       */
/*------------------------------------------------------------*/
__global__ void inject_linearborehole2d(int ix, 
										int it,
                                        float *inj,
                                        float *wfld,
										int imgj)
/*< Inject on a single plane>*/
{
  int iz = blockIdx.x*blockDim.x+threadIdx.x; // z

  if ( iz < nz_d) {
    int iaddr= iz*nx_d+ix;
    int taddr= it*nz_d+iz;
    wfld[iaddr] += inj[taddr];
  }
}

/*------------------------------------------------------------*/
/* . . Inject on a 3d vsp plane                                                       */
/*------------------------------------------------------------*/
__global__ void inject_linearborehole_3d(int ix,
					 					 int iy,
                                         int it,
                                         float *inj,
                                         float *wfld,
                                         int dat_j)
/*< Inject on a single plane>*/
{
  int iz = blockIdx.x*blockDim.x+threadIdx.x; // z
  
  if ( iz < nz_d) {
    int iaddr= iz*ny_d*nx_d+iy*nx_d+ix;
    int taddr1=  it   *nz_d+iz;
    int taddr2= (it+1)*nz_d+iz;
    
    float w1 = (float)(it%dat_j) / (float)(dat_j);
    float w2 = abs(1.f-w1);
     
    wfld[iaddr] += w1*inj[taddr1]+w2*inj[taddr2];
  }
}

/*------------------------------------------------------------*/
/* . . Inject on a vsp plane                                                       */
/*------------------------------------------------------------*/
__global__ void inject_linearborehole2d_inverse(int ix, 
										int it,
                                        float *inj,
                                        float *wfld,
                                        int imgj)
/*< Inject on a single plane>*/
{
  int iz = blockIdx.x*blockDim.x+threadIdx.x; // z

  if ( iz < nz_d) {
    int iaddr= iz*nx_d+ix;
    int taddr= it*nz_d+iz;
    wfld[iaddr] -= inj[taddr];
  }
}


/*------------------------------------------------------------*/
/* . . Extract wavefield from a 3D volume on 2D plane		  */
/*------------------------------------------------------------*/
__global__ void extract_3d(	int iz,
			  	 			float *wfld2,
			   				float *dat)
/*< Extract wavefield from a 3D volume on 2D plane>*/
{
  // . . Coordinate indexing - Global coordinates
  int ix = blockIdx.x*blockDim.x+threadIdx.x; // x
  int iy = blockIdx.y*blockDim.y+threadIdx.y; // y

  if ( ix < nx_d && iy < ny_d ) {
    int iaddr = iz*ny_d*nx_d + iy*nx_d + ix;
    int islice = iy*nx_d + ix;
    dat[islice] = wfld2[iaddr];
  }

}

/*------------------------------------------------------------*/
/* . . Cross-correlation imaging condition		      */
/*------------------------------------------------------------*/
__global__ void image_xcorr_3d(	float *p1,
								float *p2,
								float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
  	int block = nx_d*ny_d;
	int ixy = iy*nx_d+ix;
	
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = iz*block+ixy; 
      output[iaddr] += p1[iaddr]*p2[iaddr];
    }

  }
}

/*------------------------------------------------------------*/
/* . . Deconvolution Imaging condition                        */
/*------------------------------------------------------------*/
__global__ void image_decon_3d( float *p1,
                                float *p2,
                                float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
        int block = nx_d*ny_d;
        int ixy = iy*nx_d+ix;
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = iz*block+ixy;
      output[iaddr] += p1[iaddr]*p2[iaddr]/(sqrt(p1[iaddr]*p1[iaddr])*sqrt(p2[iaddr]*p2[iaddr]) + eps_d);
    }

  }
}

/*------------------------------------------------------------*/
/* . . Muting                                                 */
/*------------------------------------------------------------*/
__global__ void image_decon_3d( int souz,
                                float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
        int block = nx_d*ny_d;
        int ixy = iy*nx_d+ix;
    for(int iz=0; iz < souz; iz++){ //LOOP OVER DEPTHS TO SOURCE POINT
      int iaddr = iz*block+ixy;
      output[iaddr] += 0;
    }

  }
}

/*------------------------------------------------------------
 * Calculate distance for randomize_kernel
 *------------------------------------------------------------*/
__device__ float distance(      int x,
                                int x1,
                                int z,
                                int z1)
/*<Random number distance subsoutine>*/
{
  return sqrt((float)((x1-x)*(x1-x) + (z1-z)*(z1-z)));
}

/*------------------------------------------------------------
 * Calculate random numbers for randomize_kernel
 *------------------------------------------------------------*/
__device__ float calc(  float curr,
                        float variation,
                        float dist,
                        float border_width,
                        int in_idx)
/*<Random number generation seed>*/
{
  unsigned int seed = (unsigned int) in_idx;
  curandState s;
  curand_init(seed, 0, 0, &s);
  float rand = curand_uniform(&s);
  return curr-curr*variation*(dist/(float)border_width)*(1.-cos(SF_PI/2.*dist*rand/(float)border_width))/2.;
}


/*------------------------------------------------------------
 * Randomize velocity
 *------------------------------------------------------------*/
__global__ void randomize_kernel(float *v_d,
                                 float variation,
                                 int border_width,
                                 int nx_d,
                                 int nz_d)
/*< Randomize boundaries for minimizing spurious correlations>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iz = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  int in_idx = iz*nx_d + ix; // Linear position in current data

  int xedge=0, zedge=0;
  float dist=0.;

  if (ix > border_width && ix < nx_d-border_width && iz < nz_d-border_width)
    return;

  //Case 1, left boundary
  if (ix < border_width && iz < nz_d-border_width) {
    xedge = border_width;
    dist = distance(ix,xedge, iz, iz);
  }

  //Case 2, right boundary
  else if (ix >= nx_d-border_width && iz < nz_d-border_width) {
    xedge = nx_d-border_width-1;
    dist = distance(ix, xedge, iz,iz);
  }

  //Case 3, bottom boundary
  if (iz>=nz_d-border_width && ix >= border_width && ix <= nx_d-border_width) {
    zedge = nz_d-border_width-1;
    dist = distance(ix,ix,iz, zedge);
  }

  //Case 4, corners bottom left and right
  if (ix < border_width && iz>=nz_d-border_width) {
    xedge= border_width;
    zedge = nz_d-border_width-1;
    dist = distance(ix,xedge,iz,zedge);
  }
  if (ix >= nx_d-border_width && iz>=nz_d-border_width) {
    xedge = nx_d- border_width -1;
    zedge = nz_d - border_width -1;
    dist = distance(ix,xedge, iz,zedge);
  }

  v_d[in_idx] = calc(v_d[in_idx],variation,dist,border_width, in_idx);
}


/*------------------------------------------------------------
 * Extract wave out of gpu kernel (Only in X; Constant Z)
 *------------------------------------------------------------*/
__global__ void extract_kernel( int sou_z,
                                float* wav0_d,
                                float* data_d)
/*< Extract from a wavefield at constant sou_z depth>*/
{
  int ix = blockIdx.x*blockDim.x+threadIdx.x;
  if (ix < nx_d) {
    data_d[ix] = wav0_d[sou_z*nx_d+ix];
  }
}

/*------------------------------------------------------------
 * Extract wave out of gpu kernel (Only in X; Constant Z)
 *------------------------------------------------------------*/
__global__ void extract_kernel3d(int sou_x,
				  				 int sou_y,
                                 float* wav_d,
                                 float* data_d)
/*< Extract from a wavefield at constant sou_z depth>*/
{
  int iz = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (iz < nz_d) {
    data_d[iz] = wav_d[iz*ny_d*nx_d+sou_y*nx_d+sou_x];
  }
}



/*------------------------------------------------------------
 * Propagate_kernel, where FD is actually calculated
 *------------------------------------------------------------*/
__global__ void propagate_kernel(float *g_next,  // Input (n-1), output (n+1)
                                 float *g_curr,
                                 float *vel)
/*<Propagate wavefield with O(t^2,x^8) algorithm>*/
{
  __shared__ float s_data[BDIMY+2*RADIUS][BDIMX+2*RADIUS];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iz = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  int in_idx  = iz*nx_d + ix; // Linear position in current data
  int out_idx = in_idx; // output starts RADIUS z-slices in

  float local_curr;

  int tx = threadIdx.x + RADIUS; // Local X centre
  int tz = threadIdx.y + RADIUS; // Local Z centre

  local_curr = g_curr[in_idx];

  // Load wavefield plane into shared memory
  s_data[tz][tx] = local_curr;

  // Z Halo Direction
  if( threadIdx.y < RADIUS ) {
    if(blockIdx.y == 0) s_data[threadIdx.y][tx] = 0.f;
    else s_data[threadIdx.y][tx]= g_curr[out_idx-RADIUS*nx_d];

    if(blockIdx.y == gridDim.y -1) s_data[threadIdx.y+BDIMY+RADIUS][tx] = 0.f;
    else  s_data[threadIdx.y+BDIMY+RADIUS][tx] = g_curr[out_idx+BDIMY*nx_d];
  }

  // X Halo Direction
  if( threadIdx.x < RADIUS ){
    if(blockIdx.x==0) s_data[tz][threadIdx.x]=0.f;
    else s_data[tz][threadIdx.x]= g_curr[out_idx-RADIUS];

    if(blockIdx.x == gridDim.x -1) s_data[tz][threadIdx.x+BDIMX+RADIUS]=0.f;
    else s_data[tz][threadIdx.x+BDIMX+RADIUS] = g_curr[out_idx+BDIMX];
  }

  // Sync threads before doing FD calculation
  __syncthreads();

  // Time Derivative
  float temp = 2.f * local_curr - g_next[out_idx];

  // Spatial derivative (at centre point)
  float laplace = 2.f *cc_d[0]  * local_curr;

  // Spatial derivative (to either side)
#pragma unroll
  for( int d=1; d <= RADIUS; d++ ) {
    laplace += cc_d[d] *(s_data[tz-d][tx]+s_data[tz+d][tx]+s_data[tz][tx-d]+s_data[tz][tx+d] );
  }

  // Local velocity
  float vloc = vel[in_idx];
  // Output at each location
  g_next[out_idx] = temp + laplace * vloc * vloc;

  // Sync threads before moving on to next Z location
  __syncthreads();
}


/*------------------------------------------------------------
 * Inject wave into gpu kernel (Only in X; Constant Z)
 *------------------------------------------------------------*/
__global__ void inject_kernel_reverse_new(int sou_z,
                                int it,
                                float *inwave,
                                float *wav0_d,
                                int datj)
/*< Inject wavefield at a constant sou_z depth>*/
{
  int ix = blockIdx.x*blockDim.x+threadIdx.x;
  if (ix < nx_d) {
    int itnew = floor( (float)(it)/(float)(datj) );
    wav0_d[sou_z*nx_d+ix]+=inwave[it*nx_d+ix];
  //wav0_d[sou_z*nx_d+ix]+=(1.f-(float)(it%datj)/(float)datj)*inwave[itnew*nx_d+ix] + (float)(it%datj)/(float)datj * inwave[ (itnew-1) * nx_d+ix]  ;
  }
}


/*------------------------------------------------------------*/
/* Output image from correlating two wavefields (X and Z)     */
/*------------------------------------------------------------*/
__global__ void image_kernel(   float *p1,
                                float *p2,
                                float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iz = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  if (ix < nx_d && iz < nz_d) {
    int iaddr = iz*nx_d+ix;
    output[iaddr] += p1[iaddr]*p2[iaddr];
  }
}

/*------------------------------------------------------------*/
/* Output image from correlating two wavefields (X and Z) Decon    */
/*------------------------------------------------------------*/
__global__ void image_decon(    float *p1,
                                float *p2,
                                float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iz = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  if (ix < nx_d && iz < nz_d) {
    int iaddr = iz*nx_d+ix;
    //output[iaddr] += p1[iaddr]*p2[iaddr]/(sqrt(p1[iaddr]*p1[iaddr])*sqrt(p2[iaddr]*p2[iaddr]) + eps_d);
    output[iaddr] += p1[iaddr]*p2[iaddr]/(p1[iaddr]*p1[iaddr] + eps_d);
  }
}



/*------------------------------------------------------------ */
/*             Space derivative xig condition                  */
/*------------------------------------------------------------ */
__global__ void derivative_xig_kernel(float *vel,
                                float *swf1,
                                float *rwf1,
                                float *swf2,
                                float *rwf2,
                                float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x; /* X is ix */
  int iz = threadIdx.y + blockIdx.y * blockDim.y; /* Y is ix */

  __shared__ float swf1_data[3][BDIMT+2*NH2];  // Store thread local data - SWF curr
  __shared__ float rwf1_data[3][BDIMT+2*NH2];  // Store thread local data - RWF curr
  __shared__ float swf2_data[3][BDIMT+2*NH2];  // Store thread local data - SWF next
  __shared__ float rwf2_data[3][BDIMT+2*NH2];  // Store thread local data - RWF next

  if (ix < nx_d ) {

  int tx  = threadIdx.x +   NH2; // Local X centre
  int tx0 = threadIdx.x        ; // Local X left
  int tx1 = threadIdx.x + 2*NH2; // Local X right

  /* Zero the shared memory space */
  rwf1_data[0][tx0]=0.f; rwf1_data[0][tx1]=0.f;
  rwf2_data[0][tx0]=0.f; rwf2_data[0][tx1]=0.f;
  swf1_data[0][tx0]=0.f; swf1_data[0][tx1]=0.f;
  swf2_data[0][tx0]=0.f; swf2_data[0][tx1]=0.f;
  rwf1_data[1][tx0]=0.f; rwf1_data[1][tx1]=0.f;
  rwf2_data[1][tx0]=0.f; rwf2_data[1][tx1]=0.f;
  swf1_data[1][tx0]=0.f; swf1_data[1][tx1]=0.f;
  swf2_data[1][tx0]=0.f; swf2_data[1][tx1]=0.f;
  rwf1_data[2][tx0]=0.f; rwf1_data[2][tx1]=0.f;
  rwf2_data[2][tx0]=0.f; rwf2_data[2][tx1]=0.f;
  swf1_data[2][tx0]=0.f; swf1_data[2][tx1]=0.f;
  swf2_data[2][tx0]=0.f; swf2_data[2][tx1]=0.f;

  /* Begin otherwise loop */
  if (iz > 1 && iz < nz_d-2) {

          /* Get center of array */
          rwf1_data[0][tx]=rwf1[(iz-1)*nx_d+ix];
          rwf2_data[0][tx]=rwf2[(iz-1)*nx_d+ix];
          rwf1_data[1][tx]=rwf1[(iz  )*nx_d+ix];
          rwf2_data[1][tx]=rwf2[(iz  )*nx_d+ix];
          rwf1_data[2][tx]=rwf1[(iz+1)*nx_d+ix];
          rwf2_data[2][tx]=rwf2[(iz+1)*nx_d+ix];

          swf1_data[0][tx]=swf1[(iz-1)*nx_d+ix];
          swf2_data[0][tx]=swf2[(iz-1)*nx_d+ix];
          swf1_data[1][tx]=swf1[(iz  )*nx_d+ix];
          swf2_data[1][tx]=swf2[(iz  )*nx_d+ix];
          swf1_data[2][tx]=swf1[(iz+1)*nx_d+ix];
          swf2_data[2][tx]=swf2[(iz+1)*nx_d+ix];

          // Get boundary data
          if (tx0 < NH2) {

                /* Get Left Boundary */
                if (ix > NH2) {
                  rwf1_data[0][tx0]=rwf1[(iz-1)*nx_d+ix-NH2];
                  rwf2_data[0][tx0]=rwf2[(iz-1)*nx_d+ix-NH2];
                  rwf1_data[1][tx0]=rwf1[(iz  )*nx_d+ix-NH2];
                  rwf2_data[1][tx0]=rwf2[(iz  )*nx_d+ix-NH2];
                  rwf1_data[2][tx0]=rwf1[(iz+1)*nx_d+ix-NH2];
                  rwf2_data[2][tx0]=rwf2[(iz+1)*nx_d+ix-NH2];

                  swf1_data[0][tx0]=swf1[(iz-1)*nx_d+ix-NH2];
                  swf2_data[0][tx0]=swf2[(iz-1)*nx_d+ix-NH2];
                  swf1_data[1][tx0]=swf1[(iz  )*nx_d+ix-NH2];
                  swf2_data[1][tx0]=swf2[(iz  )*nx_d+ix-NH2];
                  swf1_data[2][tx0]=swf1[(iz+1)*nx_d+ix-NH2];
                  swf2_data[2][tx0]=swf2[(iz+1)*nx_d+ix-NH2];
                }

                /* Get Right Boundary */
            if (ix < nx_d-NH2 ) {
                  rwf1_data[0][tx+BDIMT]=rwf1[(iz-1)*nx_d+ix+BDIMT];
                  rwf2_data[0][tx+BDIMT]=rwf2[(iz-1)*nx_d+ix+BDIMT];
                  rwf1_data[1][tx+BDIMT]=rwf1[(iz  )*nx_d+ix+BDIMT];
                  rwf2_data[1][tx+BDIMT]=rwf2[(iz  )*nx_d+ix+BDIMT];
                  rwf1_data[2][tx+BDIMT]=rwf1[(iz+1)*nx_d+ix+BDIMT];
                  rwf2_data[2][tx+BDIMT]=rwf2[(iz+1)*nx_d+ix+BDIMT];

                  swf1_data[0][tx+BDIMT]=swf1[(iz-1)*nx_d+ix+BDIMT];
                  swf2_data[0][tx+BDIMT]=swf2[(iz-1)*nx_d+ix+BDIMT];
                  swf1_data[1][tx+BDIMT]=swf1[(iz  )*nx_d+ix+BDIMT];
                  swf2_data[1][tx+BDIMT]=swf2[(iz  )*nx_d+ix+BDIMT];
                  swf1_data[2][tx+BDIMT]=swf1[(iz+1)*nx_d+ix+BDIMT];
                  swf2_data[2][tx+BDIMT]=swf2[(iz+1)*nx_d+ix+BDIMT];
            }
    }
    /* END IZ=otherwise loop */

  } /* END IZ CASE */

  /* Local Velocity */
  float vloc = vel[iz*nx_d+ix];

  __syncthreads();

  if (iz < 2 || iz>nz_d-3) {

        for (int ih=0; ih<nh_d; ih++) {
          output[iz*nx_d*nh_d+ix*nh_d+ih]=0.f;
        }

  } else {

        /* Derivative imaging condition - Time and Space*/
    int nhh = (int)((float)nh_d-1.f)/2.f;
    float denom1 = 4.f*dx_d*dz_d;
    float denom2 = vloc*vloc*dt_d*dt_d;

        for (int ih=0; ih<nh_d; ih++) {

                output[iz*nx_d*nh_d+ix*nh_d+ih]+=( \
                   ((swf1_data[1][tx+1+ih-nhh]-swf1_data[1][tx-1+ih-nhh]) * \
                        (rwf1_data[1][tx+1-ih+nhh]-rwf1_data[1][tx-1-ih+nhh]) + \
                        (swf1_data[2][tx  +ih-nhh]-swf1_data[0][tx  +ih-nhh]) * \
                        (rwf1_data[2][tx  -ih+nhh]-rwf1_data[0][tx  -ih+nhh])))/ denom1 + \
                        (swf2_data[1][tx  +ih-nhh]-swf1_data[1][tx  +ih-nhh]) * \
                        (rwf1_data[1][tx  -ih+nhh]-rwf2_data[1][tx  -ih+nhh])  / denom2;

        }

  }

}
}

/*------------------------------------------------------------*/
/* Generate source illumination                               */
/*------------------------------------------------------------*/
__global__ void source_illum(float *out,
                             float *in)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iz = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  if (ix < nx_d && iz < nz_d) {
    int iaddr = iz*nx_d+ix;
    out[iaddr] += in[iaddr]*in[iaddr];
  }
}


/*------------------------------------------------------------*/
/* Generate source illumination                               */
/*------------------------------------------------------------*/
__global__ void source_illum3d(float *out,
                               float *in)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Z index

  if (ix < nx_d && iy < ny_d) {
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
    	int iaddr = iz*ny_d*nx_d+iy*nx_d+ix;
    	out[iaddr] += in[iaddr]*in[iaddr];
    }
  }
}


/*------------------------------------------------------------*/
/* . . Common Image Point Gathers                             */
/*------------------------------------------------------------*/
__global__ void cip_gpu(float *d_s, float *d_r, float *d_eimg, int nhx, int nhz, int nhy, int cipx, int cipz, int cipy, int nxpad, int nypad, int nzpad, int nb){


        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.z + blockIdx.z * blockDim.z;
        if (x < 2*nhx+1 && y < 2*nhy+1) {

        	int ihx = x - nhx;
        	int ihy = y - nhy;
        	int nxlag = 2*nhx+1;
        	int nylag = 2*nhy+1;
        	cipx += nb;
     	   	cipz += nb;
       	 	cipy += nb;
        	for (int ihz=-nhz; ihz < nhz; ihz++) {
			int z = (ihz+nzpad);
			int iaddr1 = (z*nylag*nxlag) + (y*nxlag) + x;
        		int iaddr2 = ((cipz + ihz) * nxpad * nypad) + (cipy + ihy) * nxpad + (cipx + ihx);
        		int iaddr3 = ((cipz - ihz) * nxpad * nypad) + (cipy - ihy) * nxpad + (cipx - ihx);

                /* kinetic energy (dot-product) */
                	d_eimg[iaddr1] += d_s[iaddr2] * d_r[iaddr3];

		}
	}
}
/*------------------------------------------------------------*/
/* . . Subsurface Offset Gathers                              */
/*------------------------------------------------------------*/
__global__ void sog_gpu(float *d_s, float *d_r, float *d_eimg, int nhx, int cipx, int nxpad, int nzpad, int nb){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int z = threadIdx.y + blockIdx.y * blockDim.y;

        int ihx = x - nhx;
        int nlx = 2*nhx+1;

        cipx += nb;

        if (cipx < nxpad && z < nzpad){

            // start from negative lags
            int ix1 = cipx + ihx;
            int ix2 = cipx - ihx;

            /* kinetic energy */
            d_eimg[z * nlx + x] += d_s[z * nxpad + ix1] * d_r[z * nxpad + ix2];

        }

}

/*------------------------------------------------------------*/
/* . . Image Source Spray                                     */
/*------------------------------------------------------------*/
__global__ void adj_image_sou( float *img,
                                float *rec,
                                float *sou)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
        int block = nx_d*ny_d;
        int ixy = iy*nx_d+ix;
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = iz*block+ixy;
      sou[iaddr] += img[iaddr]*rec[iaddr];
    }

  }
}

/*------------------------------------------------------------*/
/* . . Image Receiver Spray                                   */
/*------------------------------------------------------------*/
__global__ void adj_image_rec( float *img,
                                float *rec,
                                float *sou)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
        int block = nx_d*ny_d;
        int ixy = iy*nx_d+ix;
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = iz*block+ixy;
      rec[iaddr] += img[iaddr]*sou[iaddr];
    }

  }
}

 
/*-----------------------------------------------------------*/
/* . . Pad Loop                                              */
/*-----------------------------------------------------------*/
__global__ void padloop(float a, float *output)

{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index

// Need loop over Shifts
  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
        int block = nx_d*ny_d;
        int ixy = iy*nx_d+ix;
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = iz*block+ixy;
// Need to update address to have proper shift location for p1 and p2
      output[iaddr] += a;
    }

  }
}






/*------------------------------------------------------------*/
/* . . Cross-correlation imaging condition		      */
/*------------------------------------------------------------*/
__global__ void image_xcorr_3d_mpi(float *p1,
				   float *p2,
				   float *output)
/*< Zero-offset imaging condition>*/
{
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global y index
  int block = nx_d*ny_d;

  if (ix < nx_d && iy < ny_d) { //MAKE SURE IN BOUNDARY
    for(int iz=0; iz < nz_d; iz++){ //LOOP OVER ALL DEPTHS
      int iaddr = (iz+RAD)*block+iy*nx_d+ix; 
      int oaddr = iz*block+iy*nx_d+ix;
      output[oaddr] += p1[iaddr]*p2[iaddr];
    }
  }
}
/*------------------------------------------------------------*/
/* . . Inverse-scattering imaging condition				      */
/*------------------------------------------------------------*/
__global__ void image_iscat_3d(float *vel,
				float *swf1,
				float *rwf1,
				float *swf2,
				float *rwf2,
				float *output)
/*< Zero-offset imaging condition>*/
{
  __shared__ float swf1_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - SWF curr
  __shared__ float rwf1_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - RWF curr
  __shared__ float swf2_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - SWF next
  __shared__ float rwf2_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - RWF next

  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block = nx_d*ny_d;
  
  if (ix < nx_d && iy < ny_d) {

    /* Set to zero in border region */
    output[iy*nx_d+ix]=0.f;	
    
    /* Loop over all z locations */
    for (int iz=1; iz < nz_d-1; iz++){
    
      /* Compute Linear Address */
      int iaddr = iz*ny_d*nx_d+iy*nx_d+ix;
      
      /* Local Velocity */
      float vloc = vel[iaddr];
      
      /* Fill in the main region */
      /* Shift everything by +1 to allow for HALO */
      for (int id=0; id < 3; id++) {
	swf1_data[id][ty+1][tx+1]=swf1[iaddr+(id-1)*block];
	rwf1_data[id][ty+1][tx+1]=rwf1[iaddr+(id-1)*block];
	swf2_data[id][ty+1][tx+1]=swf2[iaddr+(id-1)*block];
	rwf2_data[id][ty+1][tx+1]=rwf2[iaddr+(id-1)*block];
      }
      
      /* Fill in x boundaries */
      if (tx ==       0) { swf1_data[1][ty+1][0	   ]=swf1[iaddr-1]; }
      if (tx ==       1) { rwf1_data[1][ty+1][0	   ]=rwf1[iaddr-2]; }
      if (tx ==       2) { swf2_data[1][ty+1][0	   ]=swf2[iaddr-3]; }
      if (tx ==       3) { rwf2_data[1][ty+1][0	   ]=rwf2[iaddr-4]; }
      
      if (tx == BDIMX-1) { swf1_data[1][ty+1][BDIMX+1]=swf1[iaddr+1]; }
      if (tx == BDIMX-2) { rwf1_data[1][ty+1][BDIMX+1]=rwf1[iaddr+2]; }
      if (tx == BDIMX-3) { swf2_data[1][ty+1][BDIMX+1]=swf2[iaddr+3]; }
      if (tx == BDIMX-4) { rwf2_data[1][ty+1][BDIMX+1]=rwf2[iaddr+4]; }	
      
      /* Fill in y boundaries */
      if (ty ==       0) { swf1_data[1][0	     ][tx+1]=swf1[iz*block+(iy-1)*nx_d+ix]; }
      if (ty ==       1) { rwf1_data[1][0	     ][tx+1]=rwf1[iz*block+(iy-2)*nx_d+ix]; }
      if (ty ==       2) { swf2_data[1][0	     ][tx+1]=swf2[iz*block+(iy-3)*nx_d+ix]; }
      if (ty ==       3) { rwf2_data[1][0	     ][tx+1]=rwf2[iz*block+(iy-4)*nx_d+ix]; }
      
      if (ty == BDIMY-1) { swf1_data[1][BDIMY+1][tx+1]=swf1[iz*block+(iy+1)*nx_d+ix]; }
      if (ty == BDIMY-2) { rwf1_data[1][BDIMY+1][tx+1]=rwf1[iz*block+(iy+2)*nx_d+ix]; }
      if (ty == BDIMY-3) { swf2_data[1][BDIMY+1][tx+1]=swf2[iz*block+(iy+3)*nx_d+ix]; }
      if (ty == BDIMY-4) { rwf2_data[1][BDIMY+1][tx+1]=rwf2[iz*block+(iy+4)*nx_d+ix]; }
      
      __syncthreads();
      
      /* Derivative imaging condition */
      output[iaddr]+=(						      \
		      (swf1_data[1][ty+1][tx+2]-swf1_data[1][ty+1][tx  ]) * \
		      (rwf1_data[1][ty+1][tx+2]-rwf1_data[1][ty+1][tx  ]) + \
		      (swf1_data[1][ty+2][tx+1]-swf1_data[1][ty  ][tx+1]) * \
		      (rwf1_data[1][ty+2][tx+1]-rwf1_data[1][ty  ][tx+1]) + \
		      (swf1_data[2][ty+1][tx+1]-swf1_data[0][ty+1][tx+1]) * \
		      (rwf1_data[2][ty+1][tx+1]-rwf1_data[0][ty+1][tx+1])) / \
	(4.f*dx_d*dz_d);
      
      /* Unweighted Time imaging condition */
      output[iaddr]+=
	(swf2_data[1][ty+1][tx+1]-swf1_data[1][ty+1][tx+1])*		\
	(rwf1_data[1][ty+1][tx+1]-rwf2_data[1][ty+1][tx+1])		\
	/(vloc*vloc*dt_d*dt_d);
 
     
      /* Fill in border region */
      if (ix==     0 && iy < ny_d) { output[iaddr]=0.f;}
      if (ix==nx_d-1 && iy < ny_d) { output[iaddr]=0.f;}
      if (iy==     0 && ix < nx_d) { output[iaddr]=0.f;}
      if (iy==ny_d-1 && ix < nx_d) { output[iaddr]=0.f;}
      __syncthreads();
      
    } /* End loop over z */
    
  } /* End if in nx_d and ny_d loop */
}
/*------------------------------------------------------------*/
/* . . Inverse-scattering imaging condition				      */
/*------------------------------------------------------------*/
__global__ void image_iscat_3d_mpi(float *vel,
				   float *swf1,
				   float *rwf1,
				   float *swf2,
				   float *rwf2,
				   float *output)
/*< Zero-offset inverse-scattering imaging condition>*/
{
  __shared__ float swf1_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - SWF curr
  __shared__ float rwf1_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - RWF curr
  __shared__ float swf2_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - SWF next
  __shared__ float rwf2_data[3][BDIMY+2][BDIMX+2];  // Store thread local data - RWF next
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Z index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block = nx_d*ny_d;
  
  if (ix < nx_d && iy < ny_d) {

    /* Compute Linear Address */
    int iaddr = RAD*block+iy*nx_d+ix;
    int oaddr =           iy*nx_d+ix;

    /* Fill in the main region */
    /* Shift everything by +1 to allow for HALO */
    for (int id=1; id < 3; id++) {
      swf1_data[id][ty+1][tx+1]=swf1[iaddr+(id-2)*block];
      rwf1_data[id][ty+1][tx+1]=rwf1[iaddr+(id-2)*block];
      swf2_data[id][ty+1][tx+1]=swf2[iaddr+(id-2)*block];
      rwf2_data[id][ty+1][tx+1]=rwf2[iaddr+(id-2)*block];
    }    
    
    /* Loop over all z locations */
    for (int iz=0; iz < nz_d; iz++){
          
      /* Local Velocity */
      float vloc = vel[oaddr];
      
      /* Fill in the main region */
      /* Shift everything by +1 to allow for HALO */
      for (int id=0; id < 2; id++) {
	swf1_data[id][ty+1][tx+1]=swf1_data[id+1][ty+1][tx+1];
	rwf1_data[id][ty+1][tx+1]=rwf1_data[id+1][ty+1][tx+1];
	swf2_data[id][ty+1][tx+1]=swf2_data[id+1][ty+1][tx+1];
	rwf2_data[id][ty+1][tx+1]=rwf2_data[id+1][ty+1][tx+1];
      }
      // . . Update with wavefield block one step ahead
      swf1_data[2][ty+1][tx+1]=swf1[iaddr+block];
      rwf1_data[2][ty+1][tx+1]=rwf1[iaddr+block];
      swf2_data[2][ty+1][tx+1]=swf2[iaddr+block];
      rwf2_data[2][ty+1][tx+1]=rwf2[iaddr+block];
      
      /* Fill in x boundaries */
      if (tx ==       0) { swf1_data[1][ty+1][0	     ]=swf1[iaddr-1]; }
      if (tx == BDIMX-1) { swf1_data[1][ty+1][BDIMX+1]=swf1[iaddr+1]; }
      if (tx ==       1) { rwf1_data[1][ty+1][0	     ]=rwf1[iaddr-2]; } // SHIFT TO SPREAD LOAD
      if (tx == BDIMX-2) { rwf1_data[1][ty+1][BDIMX+1]=rwf1[iaddr+2]; }
      
      /* Fill in y boundaries */
      if (ty ==       0) { swf1_data[1][0      ][tx+1]=swf1[iaddr-  nx_d]; }
      if (ty == BDIMY-1) { swf1_data[1][BDIMY+1][tx+1]=swf1[iaddr+  nx_d]; }
      if (ty ==       1) { rwf1_data[1][0      ][tx+1]=rwf1[iaddr-2*nx_d]; }
      if (ty == BDIMY-2) { rwf1_data[1][BDIMY+1][tx+1]=rwf1[iaddr+2*nx_d]; }
      
      __syncthreads();
      
      /* Derivative imaging condition */
      output[oaddr]+=(						      \
		      (swf1_data[1][ty+1][tx+2]-swf1_data[1][ty+1][tx  ]) * \
		      (rwf1_data[1][ty+1][tx+2]-rwf1_data[1][ty+1][tx  ]) + \
		      (swf1_data[1][ty+2][tx+1]-swf1_data[1][ty  ][tx+1]) * \
		      (rwf1_data[1][ty+2][tx+1]-rwf1_data[1][ty  ][tx+1]) + \
		      (swf1_data[2][ty+1][tx+1]-swf1_data[0][ty+1][tx+1]) * \
		      (rwf1_data[2][ty+1][tx+1]-rwf1_data[0][ty+1][tx+1])) / (4.f*dx_d*dz_d)+ \
	              (swf2_data[1][ty+1][tx+1]-swf1_data[1][ty+1][tx+1])*		\
	              (rwf1_data[1][ty+1][tx+1]-rwf2_data[1][ty+1][tx+1]) /(vloc*vloc*dt_d*dt_d);

      /* Roll loop index */
      iaddr+=block; oaddr+=block;

    } /* End loop over z */
  } /* End if in nx_d and ny_d loop */
}

// ------------------------------------------------------------*/
// . . Propagate Step in 2D								      */
// ------------------------------------------------------------*/
__global__ void propagate_step(float *wfld1,
			       float *wfld2,
			       float *wfld3,
			       float *vel)
{
  __shared__ float s_data[BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  
  if (ix < nx_d && iy < ny_d) {

  	int stride =  nx_d*ny_d; // Size of XY stride
  	int in_idx =  iy*nx_d + ix; // Linear position in current data
  	int out_idx = in_idx + RAD*stride; // output starts radius z-slices in
  
  	float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]
  
  	int tx = threadIdx.x + RAD; // Local X centre
  	int ty = threadIdx.y + RAD; // Local Y centre
  
#pragma unroll
  	for(int id=1; id<DIAM; id++){ // Read in local data
    	local_curr[id] = wfld2[in_idx];		
    	in_idx += stride;
 	 }
  
  	// Calculate for all Z locations in Halo
  	for(int iz=RAD; iz < nz_d-RAD; iz++){
    
    	// Advance values through registers, read wavefield
   		 for( int ij=0; ij<DIAM-1; ij++) local_curr[ij] = local_curr[ij+1];	
    
    	local_curr[DIAM-1] = wfld2[in_idx];
    
    	// Load wavefield plane into shared memory
    	s_data[ty][tx] = local_curr[RAD];
    
    	// Y Halo Direction
    	if( threadIdx.y < RAD ) {
      		s_data[threadIdx.y          ][tx] = wfld2[out_idx-  RAD*nx_d];
      		s_data[threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+BDIMY*nx_d];
    	}
    
    	// X Halo Direction
    	if( threadIdx.x < RAD ){
      		s_data[ty][threadIdx.x          ] = wfld2[out_idx-RAD  ];
      		s_data[ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+BDIMX];
    	}
    
    	// Sync threads before doing FD calculation
    	__syncthreads(); 
    
    	// Time Derivative
    	float temp = 2.f * local_curr[RAD] - wfld1[out_idx];
    
    	// Spatial derivative (at centre point)
    	float laplace = 3.f * cc2_d[0] * local_curr[RAD];
    
   	 	// Spatial derivative (to either side)
#pragma unroll
    	for( int id=1; id < COEFFS_SIZE; id++ ) {
      		laplace += \
			cc2_d[id]*(local_curr[RAD+id]+local_curr[RAD-id] + \
		  		s_data[ty-id][tx   ]+s_data[ty+id][tx   ] + \
		  		s_data[ty   ][tx-id]+s_data[ty   ][tx+id] );
    	}
    
    	// Local velocity
    	float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];
 
    	// Output at each location
   	 	wfld3[out_idx] = temp + laplace * vloc * vloc;
    
    	// Sync threads before moving on to next Z location
    	__syncthreads();
    
    	 in_idx += stride;  out_idx += stride;
  	} // End Z-loop
  }
}
// 
// ------------------------------------------------------------*/
// . . Propagate Step in 3D - Cartesian					      */
// ------------------------------------------------------------*/
// __global__ void propagate_step3d(float *wfld1,
// 			       				 float *wfld2,
// 			      				 float *wfld3,
// 			       				 float *vel)
// /*< Propagate one step in time>*/
// {
//   __shared__ float s_data[BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
//   int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
//   int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
//   int stride  = nx_d*ny_d; // Size of XY stride
//   int in_idx  = iy*nx_d + ix; // Linear position in current data
//   int out_idx = in_idx; // output starts radius z-slices in
// 
//   float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]
//   int tx = threadIdx.x + RAD; // Local X centre
//   int ty = threadIdx.y + RAD; // Local Y centre
// 
// //#pragma unroll
//   for (int id=0; id<5; id++) {
//     local_curr[id]=0.f;
//   }
//   
// //#pragma unroll
//   for(int id=RAD+1; id<DIAM; id++){ // Read in local data
//     local_curr[id] = wfld2[in_idx]; // Initialise at i z=[0,8]
//     in_idx += stride;
//   }
//   __syncthreads();
// 
//   ********************************/
//   . . LOOP OVER Z DIMENSION . . */
//   ********************************/
//   // Calculate for all Z locations in Halo
//   for(int iz=0; iz < nz_d-1; iz++){ //NEED TO TACKLE OUTSIDE OF HALO 
//     
//     // Advance values through registers, read wavefield
// //#pragma unroll
//     for( int ij=0; ij<DIAM-1; ij++) 
//       local_curr[ij] = local_curr[ij+1];	
// 
//     // . . CATCH FOR END OF Z-AXIS
//     if (iz < nz_d-5) {
//       local_curr[DIAM-1] = wfld2[in_idx]; // in_idx is 4 steps ahead
//     } else {
//       local_curr[DIAM-1] = 0.f;
//     }
//     
//     // UPDATE VALUES IN THE ID=4 PLANE
//     s_data[ty][tx] = wfld2[out_idx];
//     
//     // Y Halo Direction
//     if( threadIdx.y < RAD ) {
// 
//       if (blockIdx.y==0) s_data[threadIdx.y][tx] = 0.f;
//       else               s_data[threadIdx.y][tx] = wfld2[out_idx-RAD*nx_d];
//       if(blockIdx.y>=gridDim.y-1) s_data[threadIdx.y+BDIMY+RAD][tx] = 0.f;
//       else                        s_data[threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+BDIMY*nx_d];
//     }
// 
//     // X Halo Direction
//     if( threadIdx.x < RAD ){
//       if (blockIdx.x==0) s_data[ty][threadIdx.x] = 0.f;
//       else               s_data[ty][threadIdx.x] = wfld2[out_idx-RAD];
//       if(blockIdx.x>=gridDim.x-1) s_data[ty][threadIdx.x+BDIMX+RAD] = 0.f;
//       else                        s_data[ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+BDIMX];
//     }    
// 
//     // Sync threads before doing FD calculation
//     __syncthreads(); 
//     
//     // Time Derivative
//     float temp = 2.f * local_curr[RAD] - wfld1[out_idx];
//     
//     // Spatial derivative (at centre point)
//     float laplace = 3.f * cc2_d[0] * local_curr[RAD];
//     
//     // Spatial derivatives 
// //#pragma unroll
//     for( int id=1; id < COEFFS_SIZE; id++ ) {
//       laplace+=cc2_d[id]*(s_data[ty][tx+id]+s_data[ty+id][tx]+ \
// 			              s_data[ty][tx-id]+s_data[ty-id][tx]+		\
// 			              local_curr[RAD+id]+local_curr[RAD-id]);
//     }
//         
//     // Local velocity
//     float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];
//  
//     // Output at each location
//     wfld3[out_idx] = temp + laplace * vloc * vloc;
//     
//     // Sync threads before moving on to next Z location
//     __syncthreads();
//     
//      in_idx += stride;  out_idx += stride;
//   } // End Z-loop
//   
// }


/*------------------------------------------------------------*/
/* . . Propagate Step in 3D - Cartesian					      */
/*------------------------------------------------------------*/
__global__ void propagate_step3d(float *wfld1,
			       				 float *wfld2,
			      				 float *wfld3,
			      				 float *vel)
/*< Propagate one step in time>*/
{
  __shared__ float s_data[BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  int stride  = nx_d*ny_d; // Size of XY stride
  int in_idx  = iy*nx_d + ix; // Linear position in current data
  int out_idx = in_idx; // output starts radius z-slices in

  float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]
  int tx = threadIdx.x + RAD; // Local X centre
  int ty = threadIdx.y + RAD; // Local Y centre

#pragma unroll
  for (int id=0; id<5; id++) {
    local_curr[id]=0.f;
  }

#pragma unroll
  for(int id=RAD+1; id<DIAM; id++){ // Read in local data
    local_curr[id] = wfld2[in_idx]; // Initialise at i z=[0,8]
    in_idx += stride;
  }
  __syncthreads();

  /*********************************/
  /* . . LOOP OVER Z DIMENSION . . */
  /*********************************/
  // Calculate for all Z locations in Halo
  for(int iz=0; iz < nz_d-1; iz++){ //NEED TO TACKLE OUTSIDE OF HALO

    // Advance values through registers, read wavefield
#pragma unroll
    for( int ij=0; ij<DIAM-1; ij++)
      local_curr[ij] = local_curr[ij+1];

    // . . CATCH FOR END OF Z-AXIS
    if (iz < nz_d-5) {
      local_curr[DIAM-1] = wfld2[in_idx]; // in_idx is 4 steps ahead
    } else {
      local_curr[DIAM-1] = 0.f;
    }

    // UPDATE VALUES IN THE ID=4 PLANE
    s_data[ty][tx] = wfld2[out_idx];

    // Y Halo Direction
    if( threadIdx.y < RAD ) {
      if (blockIdx.y==0) s_data[threadIdx.y][tx] = 0.f;
      else               s_data[threadIdx.y][tx] = wfld2[out_idx-RAD*nx_d];
    }

    if( threadIdx.y > 3*RAD-1 && threadIdx.y < BDIMY) {
      if(blockIdx.y>=gridDim.y-1) s_data[threadIdx.y+8][tx] = 0.f;
      else                        s_data[threadIdx.y+8][tx] = wfld2[out_idx+RAD*nx_d];
    }

    // X Halo Direction
    if( threadIdx.x < RAD ){
      if (blockIdx.x==0) s_data[ty][threadIdx.x] = 0.f;
      else               s_data[ty][threadIdx.x] = wfld2[out_idx-RAD];
	}

    if( threadIdx.x > 3*RAD-1 && threadIdx.x < BDIMX) {
      if(blockIdx.x>=gridDim.x-1) s_data[ty][threadIdx.x+8] = 0.f;
      else                        s_data[ty][threadIdx.x+8] = wfld2[out_idx+RAD];
    }

//      if( threadIdx.y < RAD ) {
//
//       if (blockIdx.y==0) s_data[threadIdx.y][tx] = 0.f;
//       else               s_data[threadIdx.y][tx] = wfld2[out_idx-RAD*nx_d];
//
//       if(blockIdx.y>=gridDim.y-1) s_data[threadIdx.y+BDIMY+RAD][tx] = 0.f;
//       else                        s_data[threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+BDIMY*nx_d];
//     }

//     //X Halo Direction
//     if( threadIdx.x < RAD ){
//       if (blockIdx.x==0) s_data[ty][threadIdx.x] = 0.f;
//       else               s_data[ty][threadIdx.x] = wfld2[out_idx-RAD];
//
//       if(blockIdx.x>=gridDim.x-1) s_data[ty][threadIdx.x+BDIMX+RAD] = 0.f;
//       else                        s_data[ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+BDIMX];

//     }



    // Sync threads before doing FD calculation
    __syncthreads();

    // Time Derivative
    float temp = 2.f * local_curr[RAD] - wfld1[out_idx];

    // Spatial derivative (at centre point)
    float laplace = 3.f * cc2_d[0] * local_curr[RAD];

    // Spatial derivatives
#pragma unroll
    for( int id=1; id < COEFFS_SIZE; id++ ) {
      laplace+=cc2_d[id]*(s_data[ty][tx+id]+s_data[ty+id][tx]+ \
			  			  s_data[ty][tx-id]+s_data[ty-id][tx]+		\
			     		  local_curr[RAD+id]+local_curr[RAD-id]);
    }

    // Local velocity
    float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];

    // Output at each location
    wfld3[out_idx] =  temp + laplace * vloc * vloc;

    // Sync threads before moving on to next Z location
    __syncthreads();

     in_idx += stride;  out_idx += stride;
  } // End Z-loop

}


/*------------------------------------------------------------*/
/* . . Propagate Step in 3D - Cartesian	+ MPI		      */
/*------------------------------------------------------------*/
__global__ void propagate_step3d_mpi(float *wfld1,
				     				 float *wfld2,
				     				 float *wfld3,
				     				 float *vel)
{
  __shared__ float s_data[BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  int stride  = nx_d*ny_d; // Size of XY stride
  int in_idx  = iy*nx_d+ix; // Linear position in current data

  int out_idx = RAD*stride+in_idx; // output starts radius z-slices (i.e. on 4th)

  float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]
  int tx = threadIdx.x + RAD; // Local X centre
  int ty = threadIdx.y + RAD; // Local Y centre
  
  for(int id=0; id<DIAM-1; id++){ // Read in local data
    local_curr[id] = wfld2[in_idx]; // Initialise at i z=[0,8]
    in_idx += stride;
  }
  //Note in_idx is at depth level 7 at this point

  /*********************************/
  /* . . LOOP OVER Z DIMENSION . . */
  /*********************************/
  // Calculate for all Z locations in Halo
  for(int iz=0; iz < nz_d; iz++){ 
    
    // . . Bring in net wavefield
    local_curr[DIAM-1] = wfld2[in_idx]; // in_idx starts at iz=8
    
    // UPDATE VALUES IN THE ID=4 PLANE
    s_data[ty][tx] = wfld2[out_idx];
    
    // Y Halo Direction
    if( threadIdx.y < RAD ) {
      if (blockIdx.y==0) s_data[threadIdx.y][tx] = 0.f;
      else               s_data[threadIdx.y][tx] = wfld2[out_idx-RAD*nx_d];
      if(blockIdx.y>=gridDim.y-1) s_data[threadIdx.y+BDIMY+RAD][tx] = 0.f;
      else                        s_data[threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+BDIMY*nx_d];
    }
    
    // X Halo Direction
    if( threadIdx.x < RAD ){
      if (blockIdx.x==0) s_data[ty][threadIdx.x] = 0.f;
      else               s_data[ty][threadIdx.x] = wfld2[out_idx-RAD];
      if(blockIdx.x>=gridDim.x-1) s_data[ty][threadIdx.x+BDIMX+RAD] = 0.f;
      else                        s_data[ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+BDIMX];    
    }
    
    // Sync threads before doing FD calculation
     __syncthreads(); 
    
    // Time Derivative
    float temp = 2.f * local_curr[RAD] - wfld1[out_idx];
    
    // Spatial derivative (at centre point)
    float laplace = 3.f * cc2_d[0] * local_curr[RAD];
    
    // Spatial derivatives 
    for( int id=1; id < COEFFS_SIZE; id++ ) {
      laplace+=cc2_d[id]*(s_data[ty][tx+id]+s_data[ty+id][tx]+ \
			  s_data[ty][tx-id]+s_data[ty-id][tx]+		\
			  local_curr[RAD+id]+local_curr[RAD-id]);
    }
        
    // Local velocity
    float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];
 
    // Output at each location
    wfld3[out_idx] = temp + laplace * vloc * vloc;

    // Advance values through registers, read wavefield
    if (iz!=nz_d-1) {
      for( int ij=0; ij<DIAM-1; ij++) 
	local_curr[ij] = local_curr[ij+1];	
    }    

    // Sync threads before moving on to next Z location
    __syncthreads();
    
     in_idx += stride;  out_idx += stride;
  } // End Z-loop
  
}

/*------------------------------------------------------------*/
/* . . Propagate Step in 3D - Generalised				      */
/*------------------------------------------------------------*/
__global__ void propagate_step_gen(float *wfld1,
			       float *wfld2,
			       float *wfld3,
			       float *vel,
			       float *geom)
{
  __shared__ float s_data[COEFFS_SIZE][BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  int stride =  nx_d*ny_d; // Size of XY stride
  int in_idx =  iy*nx_d + ix; // Linear position in current data
  int out_idx = in_idx; // output starts radius z-slices in
  int msize = nx_d*ny_d*nz_d;

  float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]
  float local_geom[DIAM]; // Storing local geometry data for each thread at [ix,iy]

  int tx = threadIdx.x + RAD; // Local X centre
  int ty = threadIdx.y + RAD; // Local Y centre

#pragma unroll
  for (int id=0; id<5; id++) {
    local_curr[id]=0.f;
  }
  
#pragma unroll
  for(int id=RAD+1; id<DIAM; id++){ // Read in local data
    local_curr[id] = wfld2[in_idx]; // Initialise at i z=[0,8]
    in_idx += stride;
  }
  
  // Load wavefield plane into shared memory
#pragma unroll
  for (int id=0; id < 3; id++)
    s_data[id][ty][tx]=0.f;

#pragma unroll
  for (int id=3; id < COEFFS_SIZE; id++) 
    s_data[id][ty][tx] = wfld2[(id-3)*stride+iy*nx_d+ix];

  // Y Halo Direction
  if( threadIdx.y < RAD ) {
#pragma unroll
    for (int id=0; id < 3; id++) {
      s_data[id][threadIdx.y][tx] = 0.f;
      s_data[id][threadIdx.y+BDIMY+RAD][tx] = 0.f;
    }
#pragma unroll
    for (int id=3; id < COEFFS_SIZE; id++) {
      if (blockIdx.y==0) s_data[id][threadIdx.y][tx] = 0.f;
      else               s_data[id][threadIdx.y][tx] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d];

      if(blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][tx] = 0.f;
      else                        s_data[id][threadIdx.y+BDIMY+RAD][tx] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d];
    }
  }
  
    // X Halo Direction
    if( threadIdx.x < RAD ){
#pragma unroll
      for (int id=0; id < 3; id++) {
	s_data[id][ty][threadIdx.x] = 0.f;
        s_data[id][ty][threadIdx.x+BDIMX+RAD] = 0.f;
      }     
#pragma unroll
      for (int id=3; id < COEFFS_SIZE; id++) {
        if(blockIdx.x==0) s_data[id][ty][threadIdx.x] = 0.f;
        else              s_data[id][ty][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD];
        if(blockIdx.x==gridDim.x-1) s_data[id][ty][threadIdx.x+BDIMX+RAD] = 0.f;
        else                        s_data[id][ty][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMX];
      }      
    } 

    // FOUR CORNERS
    if (threadIdx.y < RAD && threadIdx.x < RAD) {
#pragma unroll
      for (int id=0; id < 3; id++) {

	s_data[id][threadIdx.y][threadIdx.x]=0.f;
	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
	s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
      }
#pragma unroll
      for (int id=3; id < COEFFS_SIZE; id++) {

	if(blockIdx.x==0 && blockIdx.y==0) s_data[id][threadIdx.y][threadIdx.x]=0.f;
	else s_data[id][threadIdx.y][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d-RAD];
	
	if(blockIdx.x==0&&blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
	else s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d-RAD];
	
	if(blockIdx.x==gridDim.x-1&&blockIdx.y==0) s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
	else s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d+BDIMX];
	
	if(blockIdx.x==gridDim.x-1&&blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
	else s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d+BDIMX];
      }
    }

    __syncthreads();

  /*********************************/
  /* . . LOOP OVER Z DIMENSION . . */
  /*********************************/
  // Calculate for all Z locations in Halo
  for(int iz=0; iz < nz_d; iz++){ //NEED TO TACKLE OUTSIDE OF HALO 
    
    // Advance values through registers, read wavefield
#pragma unroll
    for( int ij=0; ij<DIAM-1; ij++) 
      local_curr[ij] = local_curr[ij+1];	

    // . . CATCH FOR END OF Z-AXIS
    if (iz < nz_d-5) {
      local_curr[DIAM-1] = wfld2[in_idx];
    } else {
      local_curr[DIAM-1] = 0.f;
    }

    // GEOMETRY
    // Load local geometry values into memory
#pragma unroll
    for (int id=0; id < DIAM; id++) 
      local_geom[id] = geom[id*msize+iz*stride+iy*nx_d+ix];

    // ROLL ALONG VALUES IN ID=[0,3]
#pragma unroll
    for (int id=0; id < RAD; id++){
      s_data[id][ty][tx] = s_data[id+1][ty][tx];
    }
     
    // Y Halo Direction
    if( threadIdx.y < RAD ) {
#pragma unroll
      for (int id=0; id < RAD; id++) {
        s_data[id][threadIdx.y          ][tx] = s_data[id+1][threadIdx.y          ][tx];
        s_data[id][threadIdx.y+BDIMY+RAD][tx] = s_data[id+1][threadIdx.y+BDIMY+RAD][tx];
      }
    }
    
    // X Halo Direction
    if( threadIdx.x < RAD ){
#pragma unroll
      for (int id=0; id < RAD; id++) {
        s_data[id][ty][threadIdx.x          ] = s_data[id+1][ty][threadIdx.x          ];
        s_data[id][ty][threadIdx.x+BDIMX+RAD] = s_data[id+1][ty][threadIdx.x+BDIMX+RAD];      
      }
    }

    // FOUR CORNERS
    if (threadIdx.y < RAD && threadIdx.x < RAD) {
#pragma unroll
      for (int id=0; id < RAD; id++) {
		s_data[id][threadIdx.y          ][threadIdx.x          ]=s_data[id+1][threadIdx.y          ][threadIdx.x];
      	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x          ]=s_data[id+1][threadIdx.y+BDIMY+RAD][threadIdx.x];
		s_data[id][threadIdx.y          ][threadIdx.x+BDIMX+RAD]=s_data[id+1][threadIdx.y          ][threadIdx.x+BDIMX+RAD];
        s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=s_data[id+1][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD];
      }
    }

    // UPDATE VALUES IN THE ID=4 PLANE
    if (iz < nz_d-3) {
      s_data[4][ty][tx] = wfld2[out_idx+2*stride];

      // Y Halo Direction
      if( threadIdx.y < RAD ) {
    
		if (blockIdx.y==0) s_data[4][threadIdx.y][tx] = 0.f;
		else               s_data[4][threadIdx.y][tx] = wfld2[out_idx+2*stride-RAD*nx_d];

		if(blockIdx.y==gridDim.y-1) s_data[4][threadIdx.y+BDIMY+RAD][tx] = 0.f;
		else                        s_data[4][threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+2*stride+BDIMY*nx_d];
      }
    
      // X Halo Direction
      if( threadIdx.x < RAD ){
		if (blockIdx.x==0) s_data[4][ty][threadIdx.x] = 0.f;
		else               s_data[4][ty][threadIdx.x] = wfld2[out_idx+2*stride-RAD];
		if(blockIdx.x==gridDim.x-1) s_data[4][ty][threadIdx.x+BDIMX+RAD] = 0.f;
		else                        s_data[4][ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride+BDIMX];    
      }
      
      // FOUR CORNERS
      if (threadIdx.y < RAD && threadIdx.x < RAD) {
		if     (blockIdx.x==0 && blockIdx.y==0) s_data[4][threadIdx.y][threadIdx.x]=0.f;
		else s_data[4][threadIdx.y][threadIdx.x] = wfld2[out_idx+2*stride-RAD*nx_d-RAD];
		if(blockIdx.x==0 &&  blockIdx.y == gridDim.y -1) s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
		else s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x] = wfld2[out_idx+2*stride+BDIMY*nx_d-RAD];
		if(blockIdx.x==gridDim.x-1&& blockIdx.y==0) s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
		else s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride-RAD*nx_d+BDIMX];
		if(blockIdx.x==gridDim.x-1&&blockIdx.y==gridDim.y-1) s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
		else s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride+BDIMY*nx_d+BDIMX];
      }
    } else {
      // ZERO FOR BEYOND BOUNDARY
      s_data[4][ty][tx] = 0.f;

      // Y Halo Direction
      if( threadIdx.y < RAD ) {
		s_data[4][threadIdx.y][tx] = 0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][tx] = 0.f;
      }
    
      // X Halo Direction
      if( threadIdx.x < RAD ){
		s_data[4][ty][threadIdx.x] = 0.f;
		s_data[4][ty][threadIdx.x+BDIMX+RAD] = 0.f;    
      }
      
      // FOUR CORNERS
      if (threadIdx.y < RAD && threadIdx.x < RAD) {
		s_data[4][threadIdx.y][threadIdx.x]=0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
		s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
      }
    }
    // Sync threads before doing FD calculation
    __syncthreads(); 
    
    // Time Derivative
    float temp = 2.f * local_curr[RAD] - wfld1[out_idx];
    
    // Spatial derivative (at centre point)
    float laplace = cc2_d[0] * (local_geom[0]+local_geom[3]+local_geom[5]) * local_curr[RAD];
    
    // Spatial derivatives (FIRST AND SECOND ORDER)
#pragma unroll
    for( int id=1; id < COEFFS_SIZE; id++ ) {
      laplace += (cc1_d[id]*local_geom[6]+cc2_d[id]*local_geom[0])*s_data[2][ty   ][tx+id];
      laplace -= (cc1_d[id]*local_geom[6]-cc2_d[id]*local_geom[0])*s_data[2][ty   ][tx-id];
      laplace += (cc1_d[id]*local_geom[7]+cc2_d[id]*local_geom[3])*s_data[2][ty+id][tx   ];
      laplace -= (cc1_d[id]*local_geom[7]-cc2_d[id]*local_geom[3])*s_data[2][ty-id][tx   ];
      laplace += (cc1_d[id]*local_geom[8]+cc2_d[id]*local_geom[5])*local_curr[RAD+id];
      laplace -= (cc1_d[id]*local_geom[8]-cc2_d[id]*local_geom[5])*local_curr[RAD-id];
    }
    
    // . . In X-Y PLANE
    laplace+=cc12_d[0]*local_geom[1]*(s_data[2][ty+1][tx+1]+s_data[2][ty-1][tx-1]-s_data[2][ty+1][tx-1]-s_data[2][ty-1][tx+1]);
    laplace+=cc12_d[1]*local_geom[1]*(s_data[2][ty+2][tx-1]+s_data[2][ty+1][tx-2]+s_data[2][ty-2][tx+1]+s_data[2][ty-1][tx+2]);
    laplace-=cc12_d[1]*local_geom[1]*(s_data[2][ty+1][tx+2]+s_data[2][ty+2][tx+1]+s_data[2][ty-2][tx-1]+s_data[2][ty-1][tx-2]);
    laplace+=cc12_d[2]*local_geom[1]*(s_data[2][ty+2][tx+2]+s_data[2][ty-2][tx-2]-s_data[2][ty-2][tx+2]-s_data[2][ty+2][tx-2]);

    // . . IN X-Z PLANE
    laplace += cc12_d[0]*local_geom[2]*(s_data[3][ty][tx+1]+s_data[1][ty][tx-1]-s_data[3][ty][tx-1]-s_data[1][ty][tx+1]);
    laplace += cc12_d[1]*local_geom[2]*(s_data[4][ty][tx-1]+s_data[3][ty][tx-2]+s_data[0][ty][tx+1]+s_data[1][ty][tx+2]);
    laplace -= cc12_d[1]*local_geom[2]*(s_data[3][ty][tx+2]+s_data[4][ty][tx+1]+s_data[0][ty][tx-1]+s_data[1][ty][tx-2]);
    laplace += cc12_d[2]*local_geom[2]*(s_data[4][ty][tx+2]+s_data[0][ty][tx-2]-s_data[0][ty][tx+2]-s_data[4][ty][tx-2]);

    // . . IN Y-Z PLANE
    laplace += cc12_d[0]*local_geom[4]*(s_data[3][ty+1][tx]+s_data[1][ty-1][tx]-s_data[3][ty-1][tx]-s_data[1][ty+1][tx]);
    laplace += cc12_d[1]*local_geom[4]*(s_data[1][ty+2][tx]+s_data[0][ty+1][tx]+s_data[3][ty-2][tx]+s_data[4][ty-1][tx]);
    laplace -= cc12_d[1]*local_geom[4]*(s_data[4][ty+1][tx]+s_data[3][ty+2][tx]+s_data[1][ty-2][tx]+s_data[0][ty-1][tx]);
    laplace += cc12_d[2]*local_geom[4]*(s_data[4][ty+2][tx]+s_data[0][ty-2][tx]-s_data[4][ty-2][tx]-s_data[0][ty+2][tx]);
     

    // Local velocity
    float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];
 
    // Output at each location
    wfld3[out_idx] = temp + laplace * vloc * vloc;
    
    // Sync threads before moving on to next Z location
    __syncthreads();
    
     in_idx += stride;  out_idx += stride;
  } // End Z-loop
  
}

/*------------------------------------------------------------*/
/* . . Propagate Step in 3D - Crooked3d Geometry		      */
/*------------------------------------------------------------*/
__global__ void propagate_step_crooked3d(float *wfld1,
			       float *wfld2,
			       float *wfld3,
			       float *vel,
			       float *xx,
			       float *yy,
			       float *zz)
{
  __shared__ float s_data[COEFFS_SIZE][BDIMY+2*RAD][BDIMX+2*RAD];  // Store thread local data
  int ix = blockIdx.x*BDIMX + threadIdx.x; // Global X index
  int iy = blockIdx.y*BDIMY + threadIdx.y; // Global Y index
  int stride =  nx_d*ny_d; // Size of XY stride
  int in_idx =  iy*nx_d + ix; // Linear position in current data
  int out_idx = in_idx; // output starts radius z-slices in

  float local_curr[DIAM]; // Storing local data for each thread at [ix,iy]

  // GEOMETRY variables arrays
  float ig11;
  float ig12;
  float ig13,ig22,ig23,ig33,zeta1,zeta2,zeta3;
  float p1,p2,denom,e3,ae3,az0,x12y12;
  float lxx[6];
  float lyy[6];
  float lzz[6];
  
  int tx = threadIdx.x + RAD; // Local X centre
  int ty = threadIdx.y + RAD; // Local Y centre


#pragma unroll
  for (int id=0; id<5; id++) {
    local_curr[id]=0.f;
  }
  
#pragma unroll
  for(int id=RAD+1; id<DIAM; id++){ // Read in local data
    local_curr[id] = wfld2[in_idx]; // Initialise at i z=[0,8]
    in_idx += stride;
  }
  
  // Load wavefield plane into shared memory
#pragma unroll
  for (int id=0; id < 3; id++)
    s_data[id][ty][tx]=0.f;

#pragma unroll
  for (int id=3; id < COEFFS_SIZE; id++) 
    s_data[id][ty][tx] = wfld2[(id-3)*stride+iy*nx_d+ix];

  // Y Halo Direction
  if( threadIdx.y < RAD ) {
#pragma unroll
    for (int id=0; id < 3; id++) {
      s_data[id][threadIdx.y][tx] = 0.f;
      s_data[id][threadIdx.y+BDIMY+RAD][tx] = 0.f;
    }
#pragma unroll
    for (int id=3; id < COEFFS_SIZE; id++) {
      if (blockIdx.y==0) s_data[id][threadIdx.y][tx] = 0.f;
      else               s_data[id][threadIdx.y][tx] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d];

      if(blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][tx] = 0.f;
      else                        s_data[id][threadIdx.y+BDIMY+RAD][tx] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d];
    }
  }
  
    // X Halo Direction
    if( threadIdx.x < RAD ){
#pragma unroll
      for (int id=0; id < 3; id++) {
	s_data[id][ty][threadIdx.x] = 0.f;
        s_data[id][ty][threadIdx.x+BDIMX+RAD] = 0.f;
      }     
#pragma unroll
      for (int id=3; id < COEFFS_SIZE; id++) {
        if(blockIdx.x==0) s_data[id][ty][threadIdx.x] = 0.f;
        else              s_data[id][ty][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD];
        if(blockIdx.x==gridDim.x-1) s_data[id][ty][threadIdx.x+BDIMX+RAD] = 0.f;
        else                        s_data[id][ty][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMX];
      }      
    } 

    // FOUR CORNERS
    if (threadIdx.y < RAD && threadIdx.x < RAD) {
#pragma unroll
      for (int id=0; id < 3; id++) {

	s_data[id][threadIdx.y][threadIdx.x]=0.f;
	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
	s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
      }
#pragma unroll
      for (int id=3; id < COEFFS_SIZE; id++) {

	if(blockIdx.x==0 && blockIdx.y==0) s_data[id][threadIdx.y][threadIdx.x]=0.f;
	else s_data[id][threadIdx.y][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d-RAD];
	
	if(blockIdx.x==0&&blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
	else s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d-RAD];
	
	if(blockIdx.x==gridDim.x-1&&blockIdx.y==0) s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
	else s_data[id][threadIdx.y][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix-RAD*nx_d+BDIMX];
	
	if(blockIdx.x==gridDim.x-1&&blockIdx.y==gridDim.y-1) s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
	else s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD] = wfld2[(id-3)*stride+iy*nx_d+ix+BDIMY*nx_d+BDIMX];
      }
    }

  // . . Read in local geometry values xx, yy and zz
  // . . These are independent of the z-axis so only have to be initialised once per call
#pragma unroll
  for (int id=0; id<6; id++) { lxx[id]=xx[id*nx_d*ny_d+iy*nx_d+ix]; }
#pragma unroll
  for (int id=0; id<6; id++) { lyy[id]=yy[id*nx_d*ny_d+iy*nx_d+ix]; }
#pragma unroll
  for (int id=0; id<6; id++) { lzz[id]=zz[id*nx_d*ny_d+iy*nx_d+ix]; }

  // . . Geometric coefficients that are independent of e3
  // . . Helpful constants
  denom= (lyy[2]*lxx[1]-lxx[2]*lyy[1]);
  p1 = pow(lxx[1],2)+pow(lyy[1],2);
  p2 = pow(lxx[2],2)+pow(lyy[2],2);
  az0 = a_d-lzz[0];
  x12y12=lxx[2]*lxx[1]+lyy[2]*lyy[1];
  
  // . . Metric tensor coefficients  
  ig11 = (pow(lxx[2],2)+pow(lyy[2],2))/pow(denom,2);//CHECKED
  ig12 =-x12y12/pow(denom,2); //CHECKED
  ig22 = (pow(lxx[1],2)+pow(lyy[1],2))/pow(denom,2);//CHECKED
  
  // . . First-derivative terms
  zeta1= (-pow(lyy[2],3)*lxx[3]-lyy[2]*(lxx[5]*p1+ \
  		lxx[2]*(-2.f*lxx[1]*lxx[4]*2.f*lyy[1]*lyy[4]+lxx[2]*lxx[3]))+ \
  		pow(lyy[2],2)*(2.f*lyy[1]*lxx[4]+lxx[2]*lyy[3])+ \
  		lxx[2]*(lyy[5]*p1+ \
  		lxx[2]*(-2.f*lxx[1]*lyy[4]+lxx[2]*lyy[3])))/pow(denom,3);//CHECKED
  zeta2= (-lyy[5]*lxx[1]*p1+ \
  		lxx[5]*lyy[1]*p1-\
  		2.f*lxx[2]*lxx[1]*lyy[1]*lxx[4]-2.f*lyy[2]*pow(lyy[1],2)*lxx[4]+ \
  		2.f*lxx[2]*pow(lxx[1],2)*lyy[4]+2.f*lyy[2]*lxx[1]*lyy[1]*lyy[4]+ \
  		pow(lxx[2],2)*lyy[1]*lxx[3]+pow(lyy[2],2)*lyy[1]*lxx[3]- \
  		p2*lxx[1]*lyy[3])/pow(denom,3);//CHECKED
  
  __syncthreads();

  /*********************************/
  /* . . LOOP OVER Z DIMENSION . . */
  /*********************************/
  // Calculate for all Z locations in Halo
  for(int iz=0; iz < nz_d; iz++){ //NEED TO TACKLE OUTSIDE OF HALO 
    
    // Advance values through registers, read wavefield
#pragma unroll
    for( int ij=0; ij<DIAM-1; ij++) 
      local_curr[ij] = local_curr[ij+1];	

    // . . CATCH FOR END OF Z-AXIS
    if (iz < nz_d-5) {
      local_curr[DIAM-1] = wfld2[in_idx];
    } else {
      local_curr[DIAM-1] = 0.f;
    }

    // ROLL ALONG VALUES IN ID=[0,3]
#pragma unroll
    for (int id=0; id < RAD; id++){
      s_data[id][ty][tx] = s_data[id+1][ty][tx];
    }
     
    // Y Halo Direction
    if( threadIdx.y < RAD ) {
#pragma unroll
      for (int id=0; id < RAD; id++) {
        s_data[id][threadIdx.y          ][tx] = s_data[id+1][threadIdx.y          ][tx];
        s_data[id][threadIdx.y+BDIMY+RAD][tx] = s_data[id+1][threadIdx.y+BDIMY+RAD][tx];
      }
    }
    
    // X Halo Direction
    if( threadIdx.x < RAD ){
#pragma unroll
      for (int id=0; id < RAD; id++) {
        s_data[id][ty][threadIdx.x          ] = s_data[id+1][ty][threadIdx.x          ];
        s_data[id][ty][threadIdx.x+BDIMX+RAD] = s_data[id+1][ty][threadIdx.x+BDIMX+RAD];      
      }
    }

    // FOUR CORNERS
    if (threadIdx.y < RAD && threadIdx.x < RAD) {
#pragma unroll
      for (int id=0; id < RAD; id++) {
		s_data[id][threadIdx.y          ][threadIdx.x          ]=s_data[id+1][threadIdx.y          ][threadIdx.x];
      	s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x          ]=s_data[id+1][threadIdx.y+BDIMY+RAD][threadIdx.x];
		s_data[id][threadIdx.y          ][threadIdx.x+BDIMX+RAD]=s_data[id+1][threadIdx.y          ][threadIdx.x+BDIMX+RAD];
        s_data[id][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=s_data[id+1][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD];
      }
    }

    // UPDATE VALUES IN THE ID=4 PLANE
    if (iz < nz_d-3) {
      s_data[4][ty][tx] = wfld2[out_idx+2*stride];

      // Y Halo Direction
      if( threadIdx.y < RAD ) {
    
		if (blockIdx.y==0) s_data[4][threadIdx.y][tx] = 0.f;
		else               s_data[4][threadIdx.y][tx] = wfld2[out_idx+2*stride-RAD*nx_d];

		if(blockIdx.y==gridDim.y-1) s_data[4][threadIdx.y+BDIMY+RAD][tx] = 0.f;
		else                        s_data[4][threadIdx.y+BDIMY+RAD][tx] = wfld2[out_idx+2*stride+BDIMY*nx_d];
      }
    
      // X Halo Direction
      if( threadIdx.x < RAD ){
		if (blockIdx.x==0) s_data[4][ty][threadIdx.x] = 0.f;
		else               s_data[4][ty][threadIdx.x] = wfld2[out_idx+2*stride-RAD];
		if(blockIdx.x==gridDim.x-1) s_data[4][ty][threadIdx.x+BDIMX+RAD] = 0.f;
		else                        s_data[4][ty][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride+BDIMX];    
      }
      
      // FOUR CORNERS
      if (threadIdx.y < RAD && threadIdx.x < RAD) {
		if     (blockIdx.x==0 && blockIdx.y==0) s_data[4][threadIdx.y][threadIdx.x]=0.f;
		else s_data[4][threadIdx.y][threadIdx.x] = wfld2[out_idx+2*stride-RAD*nx_d-RAD];
		if(blockIdx.x==0 &&  blockIdx.y == gridDim.y -1) s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
		else s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x] = wfld2[out_idx+2*stride+BDIMY*nx_d-RAD];
		if(blockIdx.x==gridDim.x-1&& blockIdx.y==0) s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
		else s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride-RAD*nx_d+BDIMX];
		if(blockIdx.x==gridDim.x-1&&blockIdx.y==gridDim.y-1) s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
		else s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD] = wfld2[out_idx+2*stride+BDIMY*nx_d+BDIMX];
      }
    } else {
      // ZERO FOR BEYOND BOUNDARY
      s_data[4][ty][tx] = 0.f;

      // Y Halo Direction
      if( threadIdx.y < RAD ) {
		s_data[4][threadIdx.y][tx] = 0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][tx] = 0.f;
      }
    
      // X Halo Direction
      if( threadIdx.x < RAD ){
		s_data[4][ty][threadIdx.x] = 0.f;
		s_data[4][ty][threadIdx.x+BDIMX+RAD] = 0.f;    
      }
      
      // FOUR CORNERS
      if (threadIdx.y < RAD && threadIdx.x < RAD) {
		s_data[4][threadIdx.y][threadIdx.x]=0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x]=0.f;
		s_data[4][threadIdx.y][threadIdx.x+BDIMX+RAD]=0.f;
		s_data[4][threadIdx.y+BDIMY+RAD][threadIdx.x+BDIMX+RAD]=0.f;
      }
    }
    
    // . . Calculate geometry section for coefficients that depend on e3
    e3 = a_d*(float)(iz)/((float)(nz_d-1));
    ae3 = a_d-e3;
    ig13 = (a_d-e3)*(-lzz[2]*x12y12+p2*lzz[1])/(az0*pow(denom,2)); //CHECKED
    ig23 = (a_d-e3)*(-lzz[1]*x12y12+p1*lzz[2])/(az0*pow(denom,2)); //CHECKED
    ig33 = ( pow(az0,2)*(-pow((x12y12+pow(a_d*ae3,2)*lzz[2]*lzz[1]/pow(az0,4)),2)+ \
    	(p2+pow(a_d*ae3*lzz[2],2)/pow(az0,4)* \
    	(p1+pow(a_d*ae3*lzz[1],2)/pow(az0,4)))))/pow(a_d*denom,2);//CHECKED
    zeta3 = (ae3*(lzz[2]*(-lyy[5]*lxx[1]*p1+lxx[5]*lyy[1]* \
    	p1-2.f*lxx[2]*lxx[1]*lyy[1]*lxx[4]- \
    	2.f*lyy[2]*pow(lyy[1],2)*lxx[4]+2.f*lxx[2]*pow(lxx[1],2)*lyy[4]+ \
    	2.f*lyy[2]*lxx[1]*lyy[1]*lyy[4]+pow(lxx[2],2)*lyy[1]*lxx[3]+ \
    	pow(lyy[2],2)*lyy[1]*lxx[3]-p2*lxx[1]*lyy[3])+ \
    	pow(lyy[2],3)*(-lzz[1]*lxx[3]+lxx[1]*lzz[3])+ \
    	pow(lyy[2],2)*(lxx[2]*lzz[1]*lyy[3]+ \
    	lyy[1]*(2.f*lzz[1]*lxx[4]-2.f*lxx[1]*lzz[4]-lxx[2]*lzz[3])) + \
    	lyy[2]*(lzz[5]*lxx[1]*p1-lxx[5]* \
    	p1*lzz[1]+lxx[2]*(-2.f*lyy[1]*lzz[1]*lyy[4]- \
    	2.f*pow(lxx[1],2)*lzz[4]+2.f*pow(lyy[1],2)*lzz[4]-lxx[2]*lzz[1]*lxx[3]+ \
    	lxx[1]*(2.f*lzz[1]*lxx[4]+lxx[2]*lzz[3])))+ \
    	lxx[2]*(-lzz[5]*lyy[1]*p1+ \
    	lyy[5]*p1*lzz[1]+ \
    	lxx[2]*(lxx[1]*(-2.f*lzz[1]*lyy[4]+2.f*lyy[1]*lzz[4])+ \
    	lxx[2]*(lzz[1]*lyy[3]-lyy[1]*lzz[3])))))/ \
    	(az0*pow(denom,3));    
    
    // Sync threads before doing FD calculation
    __syncthreads(); 
    
    // Time Derivative
    float temp = 2.f * local_curr[RAD] - wfld1[out_idx];
    
    // Spatial derivative (at centre point)
    float laplace = cc2_d[0] * (ig11+ig22+ig33) * local_curr[RAD];
    
    // Spatial derivatives (FIRST AND SECOND ORDER)
#pragma unroll
    for( int id=1; id < COEFFS_SIZE; id++ ) {
      laplace += (cc1_d[id]*zeta1+cc2_d[id]*ig11)*s_data[2][ty   ][tx+id];
      laplace -= (cc1_d[id]*zeta1-cc2_d[id]*ig11)*s_data[2][ty   ][tx-id];
      laplace += (cc1_d[id]*zeta2+cc2_d[id]*ig22)*s_data[2][ty+id][tx   ];
      laplace -= (cc1_d[id]*zeta2-cc2_d[id]*ig22)*s_data[2][ty-id][tx   ];
      laplace += (cc1_d[id]*zeta3+cc2_d[id]*ig33)*local_curr[RAD+id];
      laplace -= (cc1_d[id]*zeta3-cc2_d[id]*ig33)*local_curr[RAD-id];
    }
    
    // . . In X-Y PLANE
    laplace +=cc12_d[0]*ig12*(s_data[2][ty+1][tx+1]+s_data[2][ty-1][tx-1]-s_data[2][ty+1][tx-1]-s_data[2][ty-1][tx+1]);
    laplace +=cc12_d[1]*ig12*(s_data[2][ty+2][tx-1]+s_data[2][ty+1][tx-2]+s_data[2][ty-2][tx+1]+s_data[2][ty-1][tx+2]);
    laplace -=cc12_d[1]*ig12*(s_data[2][ty+1][tx+2]+s_data[2][ty+2][tx+1]+s_data[2][ty-2][tx-1]+s_data[2][ty-1][tx-2]);
    laplace +=cc12_d[2]*ig12*(s_data[2][ty+2][tx+2]+s_data[2][ty-2][tx-2]-s_data[2][ty-2][tx+2]-s_data[2][ty+2][tx-2]);

    // . . IN X-Z PLANE
    laplace += cc12_d[0]*ig13*(s_data[3][ty][tx+1]+s_data[1][ty][tx-1]-s_data[3][ty][tx-1]-s_data[1][ty][tx+1]);
    laplace += cc12_d[1]*ig13*(s_data[4][ty][tx-1]+s_data[3][ty][tx-2]+s_data[0][ty][tx+1]+s_data[1][ty][tx+2]);
    laplace -= cc12_d[1]*ig13*(s_data[3][ty][tx+2]+s_data[4][ty][tx+1]+s_data[0][ty][tx-1]+s_data[1][ty][tx-2]);
    laplace += cc12_d[2]*ig13*(s_data[4][ty][tx+2]+s_data[0][ty][tx-2]-s_data[0][ty][tx+2]-s_data[4][ty][tx-2]);

    // . . IN Y-Z PLANE
    laplace += cc12_d[0]*ig23*(s_data[3][ty+1][tx]+s_data[1][ty-1][tx]-s_data[3][ty-1][tx]-s_data[1][ty+1][tx]);
    laplace += cc12_d[1]*ig23*(s_data[1][ty+2][tx]+s_data[0][ty+1][tx]+s_data[3][ty-2][tx]+s_data[4][ty-1][tx]);
    laplace -= cc12_d[1]*ig23*(s_data[4][ty+1][tx]+s_data[3][ty+2][tx]+s_data[1][ty-2][tx]+s_data[0][ty-1][tx]);
    laplace += cc12_d[2]*ig23*(s_data[4][ty+2][tx]+s_data[0][ty-2][tx]-s_data[4][ty-2][tx]-s_data[0][ty+2][tx]);
     

    // Local velocity
    float vloc = vel[iz*ny_d*nx_d+iy*nx_d+ix];
 
    // Output at each location
    wfld3[out_idx] = temp + laplace * vloc * vloc;
    
    // Sync threads before moving on to next Z location
    __syncthreads();
    
     in_idx += stride;  out_idx += stride;
  } // End Z-loop
  
}


/*------------------------------------------------------------*/
/* . . Apply DABC on 1st axis                                         */
/*------------------------------------------------------------*/
//__global__ void apply_dabc1axis(float *awfld1,
 //                                    float *awfld2,
  //                                   float *awfld3,
//                                     float *awfld1A,
//                                     float *awfld2A,
//                                     float *awfld3A,
//                                     float *wfld3,
//                                     float *vel)
//{
        /* for left face */
        /* awfld's are size(nz,ndab,norder) */
//    int iz =blockIdx.x*blockDim.x+threadIdx.x; // Global y index
//        int block1=ndab_d*p_d;
 //       int block2=p_d;
 //       int block3=nx_d;
//        float dtdh=(dt_d/dh_d);
//        int iaddr,iaddrp1,iaddrp2,iaddrd1,iaddrd2;
//        int vaddr,vaddrA;
//        float vdxdt2,vdxdt2A;
//        float v,v2,vd,vd2,w,w2;

        //perform wave equation in boundary
//        if (iz < nz_d-1 && iz > 0) {
//                        for(int idab=1; idab < ndab_d-1; idab++){
//                                vaddr=iz*block3+(idab-1);
//                                vaddrA=iz*block3+(nx_d-idab);
//                                vdxdt2 = (vel[vaddr] *vel[vaddr]) *dtdh*dtdh;
//                                vdxdt2A= (vel[vaddrA]*vel[vaddrA])*dtdh*dtdh;
//                                for(int ip=0; ip< p_d; ip++){
//                                        iaddr=iz*block1+idab*block2+ip;
//
                                        //front face
//                                        awfld3[iaddr]=2.f *awfld2[iaddr]-awfld1[iaddr] +\
                                                        vdxdt2*(-4.f*awfld2[iaddr] +\
                                                    awfld2[(iz+1)*block1+idab*block2+ip]+awfld2[(iz-1)*block1+idab*block2+ip]+\
                                                    awfld2[iz*block1+(idab+1)*block2+ip]+awfld2[iz*block1+(idab-1)*block2+ip]);

                                        //back face
//                                        awfld3A[iaddr]=2.f *awfld2A[iaddr]-awfld1A[iaddr] +\
                                                        vdxdt2A*(-4.f*awfld2A[iaddr] +\
                                                    awfld2A[(iz+1)*block1+idab*block2+ip]+awfld2A[(iz-1)*block1+idab*block2+ip]+\
                                                    awfld2A[iz*block1+(idab+1)*block2+ip]+awfld2A[iz*block1+(idab-1)*block2+ip]);
//                                }
//                        }
//        }

        //set edge of DAB to last interior grid points
//        if (iz < nz_d ) {
                        //front face
//                        awfld3[iz*block1+0*block2+0]=wfld3[iz*block3+4];
                        //back face
//                        awfld3A[iz*block1+0*block2+0]=wfld3[iz*block3+(nx_d-5)];
//        }

        //go up the "ladder"
//        if (iz < nz_d) {
//                vaddr=iz*block3+4;
//                vaddrA=iz*block3+(nx_d-5);
//                vd  = (vel[vaddr]*vel[vaddr])*dtdh;
//                vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
//                for(int ip=0; ip< p_d-1; ip++){
//                        iaddr  =iz*block1+0*block2+(ip+1);
//                        iaddrp1=iz*block1+0*block2+(ip+1);
//                        iaddrp2=iz*block1+1*block2+(ip+1);
//                        iaddrd1=iz*block1+0*block2+ ip   ;
//                        iaddrd2=iz*block1+1*block2+ ip   ;
                        //front face
//                        awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\
                                              (1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\
                                             (-1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\
                                             (-1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\
                                             (-1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\
                                              (1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\
                                              (1.f-vd)/(1.f+vd)*awfld3[iaddrd1];
                        //back face
//                        awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
                                               (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
                                              (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
                                              (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
                                              (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
                                               (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
                                               (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrd1];
//                }
//        }

        // apply termination condition
//        if (iz < nz_d) {
//                vaddr=iz*block3+0;
//                vaddrA=iz*block3+(nx_d-1);
//                v =vel[vaddr];
//                v2=vel[vaddrA];
//                w = 1.f/dt_d+v/dh_d;
//                w2= 1.f/dt_d+v2/dh_d;
//                iaddr  =iz*block1+(ndab_d-1)*block2+(p_d-1);
//                iaddrp2=iz*block1+(ndab_d-2)*block2+(p_d-1);
                //front face
//                awfld3[iaddr]=((awfld2[iaddrp2] - awfld3[iaddrp2] + awfld2[iaddr])/dt_d +\
                                        v*(awfld2[iaddrp2] - awfld2[iaddr]   + awfld3[iaddrp2])/dh_d)/w;
                //back face
//                awfld3A[iaddr]=((awfld2A[iaddrp2] - awfld3A[iaddrp2] + awfld2A[iaddr])/dt_d +\
                                        v2*(awfld2A[iaddrp2] - awfld2A[iaddr]   + awfld3A[iaddrp2])/dh_d)/w2;
//        }

        // go down the "ladder"
//        if (iz < nz_d) {
//                vaddr=iz*block3+0;
//                vaddrA=iz*block3+(nx_d-1);
//                vd  = (vel[vaddr]*vel[vaddr])*dtdh;
 //               vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
//                for(int ip=p_d-2; ip>= 0; ip--){
//                        iaddr  =iz*block1+(ndab_d-1)*block2+ ip   ;
//                        iaddrp1=iz*block1+(ndab_d-1)*block2+(ip+1);
//                        iaddrp2=iz*block1+(ndab_d-2)*block2+(ip+1);
//                        iaddrd1=iz*block1+(ndab_d-1)*block2+ ip   ;
//                        iaddrd2=iz*block1+(ndab_d-2)*block2+ ip   ;
                        //front face
//                        awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\
                                                  (1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\
                                                 (-1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\
                                                 (-1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\
                                                 (-1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\
                                                  (1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\
                                                  (1.f-vd)/(1.f+vd)*awfld3[iaddrp1];
                        //bacl face
//                        awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
                                                   (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
                                                  (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
                                                  (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
                                                  (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
                                           (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
                                                   (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrp1];
//                }
//        }
        //replace halo points with DAB points
//        if (iz < nz_d ) {
                        //front face
//                        wfld3[iz*block3+0]=awfld3[iz*block1+4*block2];
//                        wfld3[iz*block3+1]=awfld3[iz*block1+3*block2];
//                        wfld3[iz*block3+2]=awfld3[iz*block1+2*block2];
//                        wfld3[iz*block3+3]=awfld3[iz*block1+1*block2];
                        //back face
//                        wfld3[iz*block3+(nx_d-1)]=awfld3A[iz*block1+4*block2];
 //                       wfld3[iz*block3+(nx_d-2)]=awfld3A[iz*block1+3*block2];
//                        wfld3[iz*block3+(nx_d-3)]=awfld3A[iz*block1+2*block2];
//                        wfld3[iz*block3+(nx_d-4)]=awfld3A[iz*block1+1*block2];
//        }

//}

/*------------------------------------------------------------*/
/* . . Apply DABC on top                              */
/*------------------------------------------------------------*/
//__global__ void apply_dabctop(float *awfld1,
//                                     float *awfld2,
//                                     float *awfld3,
//                                     float *wfld3,
//                                     float *vel)
//{
        /* for left face */
        /* awfld's are size(nz,ndab,norder) */
//    int ix =blockIdx.x*blockDim.x+threadIdx.x; // Global y index
//        int block1=ndab_d*p_d;
//        int block2=p_d;
//        int block3=nx_d;
//        float dtdh=(dt_d/dh_d);
//        int iaddr,iaddrp1,iaddrp2,iaddrd1,iaddrd2;
//        int vaddr,vaddrA;
 //       float vdxdt2,vdxdt2A;
//        float v,v2,vd,vd2,w,w2;

        //perform wave equation in boundary
//        if (ix < nx_d-1 && ix > 0) {
//                        for(int idab=1; idab < ndab_d-1; idab++){
//                                vaddr=(idab-1)*block3+ix;
//                              vaddrA=(nz_d-idab)*block3+ix;
//                                vdxdt2 = (vel[vaddr] *vel[vaddr]) *dtdh*dtdh;
//                              vdxdt2A= (vel[vaddrA]*vel[vaddrA])*dtdh*dtdh;
//                                for(int ip=0; ip< p_d; ip++){
//                                        iaddr=ix*block1+idab*block2+ip;

                                        //front face
//                                        awfld3[iaddr]=2.f *awfld2[iaddr]-awfld1[iaddr] +\
                                                        vdxdt2*(-4.f*awfld2[iaddr] +\
                                                    awfld2[(ix+1)*block1+idab*block2+ip]+awfld2[(ix-1)*block1+idab*block2+ip]+\
                                                    awfld2[ix*block1+(idab+1)*block2+ip]+awfld2[ix*block1+(idab-1)*block2+ip]);


                                        //back face
//                                      awfld3A[iaddr]=2.f *awfld2A[iaddr]-awfld1A[iaddr] +\
//                                                      vdxdt2A*(-4.f*awfld2A[iaddr] +\
//                                                  awfld2A[(iz+1)*block1+idab*block2+ip]+awfld2A[(iz-1)*block1+idab*block2+ip]+\
//                                                  awfld2A[iz*block1+(idab+1)*block2+ip]+awfld2A[iz*block1+(idab-1)*block2+ip]);
//                                }
//                        }
//        }

        //set edge of DAB to last interior grid points
//        if (ix < nx_d ) {
                        //front face
//                        awfld3[ix*block1+0*block2+0]=wfld3[4*block3+ix];
                        //back face
//                      awfld3A[ix*block1+0*block2+0]=wfld3[(nz_d-5)*block3+ix];
//        }

        //go up the "ladder"
//        if (ix < nx_d) {
//                vaddr=4*block3+ix;
//              vaddrA=(nz_d-5)*block3+ix;
//                vd  = (vel[vaddr]*vel[vaddr])*dtdh;
//              vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
//                for(int ip=0; ip< p_d-1; ip++){
//                        iaddr  =ix*block1+0*block2+(ip+1); //(ip+1,1,i2)
//                        iaddrp1=ix*block1+0*block2+(ip+1);// (ip+1,1,i2)
//                        iaddrp2=ix*block1+1*block2+(ip+1); //(ip+1,2,i2)
//                        iaddrd1=ix*block1+0*block2+ ip   ; //(ip  ,1,i2)
//                        iaddrd2=ix*block1+1*block2+ ip   ; //(ip  ,2,i2)
                        //front face
//                        awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\ //(ip+1,1,i2)
//                                              (1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\ //(ip+1,2,i2)
 //                                            (-1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\ //(ip  ,2,i2)
//                                             (-1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\// (ip  ,1,i2)
//                                             (-1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\// (ip+1,2,i2)
//                                              (1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\// (ip  ,2,i2)
//                                              (1.f-vd)/(1.f+vd)*awfld3[iaddrd1];  //(ip  ,1,i2)
                        //back face
                        // awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
//                                             (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
//                                            (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
//                                            (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
//                                            (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
//                                             (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
//                                             (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrd1];
//                }
//        }

        // apply termination condition
//        if (ix < nx_d) {
//                vaddr=0*block3+ix;
//              vaddrA=(nz_d-1)*block3+ix;
//                v =vel[vaddr];
//              v2=vel[vaddrA];
//                w = 1.f/dt_d+v/dh_d;
//                w2= 1.f/dt_d+v2/dh_d;
//                iaddr  =ix*block1+(ndab_d-1)*block2+(p_d-1);
//                iaddrp2=ix*block1+(ndab_d-2)*block2+(p_d-1);

                //front face
//                awfld3[iaddr]=((awfld2[iaddrp2] - awfld3[iaddrp2] + awfld2[iaddr])/dt_d +\
                                        v*(awfld2[iaddrp2] - awfld2[iaddr]   + awfld3[iaddrp2])/dh_d)/w;
                //back face
                // awfld3A[iaddr]=((awfld2A[iaddrp2] - awfld3A[iaddrp2] + awfld2A[iaddr])/dt_d +\
//                                      v2*(awfld2A[iaddrp2] - awfld2A[iaddr]   + awfld3A[iaddrp2])/dh_d)/w2;
 //       }

        // go down the "ladder"
 //       if (ix < nx_d) {
 //               vaddr=0*block3+ix;
//              vaddrA=(nz_d-1)*block3+ix;
 //               vd  = (vel[vaddr]*vel[vaddr])*dtdh;
//              vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
 //               for(int ip=p_d-2; ip>= 0; ip--){
//                        iaddr  =ix*block1+(ndab_d-1)*block2+ ip   ; //(ip  ,nadb,i2)
//                        iaddrp1=ix*block1+(ndab_d-1)*block2+(ip+1); //(ip+1,nadb,i2)
//                        iaddrp2=ix*block1+(ndab_d-2)*block2+(ip+1); //(ip+1,nadb-1,i2)
//                        iaddrd1=ix*block1+(ndab_d-1)*block2+ ip   ; //(ip  ,nadb,i2)
//                        iaddrd2=ix*block1+(ndab_d-2)*block2+ ip   ; //(ip  ,nadb-1,i2)

                        //front face
//                        awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\ //(ip  ,nadb,i2)
//                                                  (1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\ //(ip  ,nadb-1,i2)
//                                                 (-1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\ //(ip+1,nadb-1,i2)
//                                                 (-1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\ //(ip+1,nadb,i2)
//                                                 (-1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\ //(ip  ,nadb-1,i2)
//                                                  (1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\ //(ip+1,nadb-1,i2)
//                                                  (1.f-vd)/(1.f+vd)*awfld3[iaddrp1];  //(ip+1,nadb,i2)
                        //bacl face
                        // awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
//                                                 (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
//                                                (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
//                                                (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
//                                                (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
//                                         (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
//                                                 (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrp1];
//                }
//        }


        //replace halo points with DAB points
//        if (ix < nx_d ) {
                        //front face
//                        wfld3[0*block3+ix]=awfld3[ix*block1+4*block2];
 //                       wfld3[1*block3+ix]=awfld3[ix*block1+3*block2];
//                        wfld3[2*block3+ix]=awfld3[ix*block1+2*block2];
//                        wfld3[3*block3+ix]=awfld3[ix*block1+1*block2];
                        //back face
//                      wfld3[(nx_d-1)*block3+ix]=awfld3A[ix*block1+4*block2];
//                      wfld3[(nx_d-2)*block3+ix]=awfld3A[ix*block1+3*block2];
//                      wfld3[(nx_d-3)*block3+ix]=awfld3A[ix*block1+2*block2];
//                      wfld3[(nx_d-4)*block3+ix]=awfld3A[ix*block1+1*block2];
//        }

//}

/*------------------------------------------------------------*/
/* . . Apply DABC on bottom                                   */
/*------------------------------------------------------------*/
//__global__ void apply_dabcbottom(float *awfld1A,
//                                     float *awfld2A,
//                                     float *awfld3A,
//                                     float *wfld3,
//                                     float *vel)
//{
        /* for left face */
        /* awfld's are size(nz,ndab,norder) */
//    int ix =blockIdx.x*blockDim.x+threadIdx.x; // Global y index
//        int block1=ndab_d*p_d;
//        int block2=p_d;
//        int block3=nx_d;
//        float dtdh=(dt_d/dh_d);
//        int iaddr,iaddrp1,iaddrp2,iaddrd1,iaddrd2;
//        int vaddr,vaddrA;
//        float vdxdt2,vdxdt2A;
//        float v,v2,vd,vd2,w,w2;

        //perform wave equation in boundary
//        if (ix < nx_d-1 && ix > 0) {
//                        for(int idab=1; idab < ndab_d-1; idab++){
//                              vaddr=(idab-1)*block3+ix;
//                                vaddrA=(nz_d-idab)*block3+ix;
//                              vdxdt2 = (vel[vaddr] *vel[vaddr]) *dtdh*dtdh;
//                                vdxdt2A= (vel[vaddrA]*vel[vaddrA])*dtdh*dtdh;
//                                for(int ip=0; ip< p_d; ip++){
//                                        iaddr=ix*block1+idab*block2+ip;

                                        //front face
//                                      awfld3[iaddr]=2.f *awfld2[iaddr]-awfld1[iaddr] +\
//                                                      vdxdt2*(-4.f*awfld2[iaddr] +\
//                                                  awfld2[(ix+1)*block1+idab*block2+ip]+awfld2[(ix-1)*block1+idab*block2+ip]+\
//                                                  awfld2[ix*block1+(idab+1)*block2+ip]+awfld2[ix*block1+(idab-1)*block2+ip]);

                                        //back face
//                                        awfld3A[iaddr]=2.f *awfld2A[iaddr]-awfld1A[iaddr] +\
                                                        vdxdt2A*(-4.f*awfld2A[iaddr] +\
                                                    awfld2A[(ix+1)*block1+idab*block2+ip]+awfld2A[(ix-1)*block1+idab*block2+ip]+\
                                                    awfld2A[ix*block1+(idab+1)*block2+ip]+awfld2A[ix*block1+(idab-1)*block2+ip]);
//                                }
 //                       }
//        }

        //set edge of DAB to last interior grid points
//        if (ix < nx_d ) {
                        //front face
//                      awfld3[ix*block1+0*block2+0]=wfld3[4*block3+ix];
                        //back face
//                        awfld3A[ix*block1+0*block2+0]=wfld3[(nz_d-5)*block3+ix];
//        }

        //go up the "ladder"
//        if (ix < nx_d) {
//              vaddr=4*block3+ix;
//                vaddrA=(nz_d-5)*block3+ix;
//              vd  = (vel[vaddr]*vel[vaddr])*dtdh;
//                vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
//                for(int ip=0; ip< p_d-1; ip++){
//                        iaddr  =ix*block1+0*block2+(ip+1);
//                        iaddrp1=ix*block1+0*block2+(ip+1);
//                        iaddrp2=ix*block1+1*block2+(ip+1);
//                        iaddrd1=ix*block1+0*block2+ ip   ;
//                        iaddrd2=ix*block1+1*block2+ ip   ;
                        //front face
//                      awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\
//                                            (1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\
//                                           (-1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\
//                                           (-1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\
//                                           (-1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\
//                                            (1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\
//                                            (1.f-vd)/(1.f+vd)*awfld3[iaddrd1];
                        //back face
//                        awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
//                                               (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
                                              (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
                                              (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
                                              (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
                                               (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
                                               (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrd1];
//                }
//        }

        // apply termination condition
//        if (ix < nx_d) {
//              vaddr=0*block3+ix;
//                vaddrA=(nz_d-1)*block3+ix;
//              v =vel[vaddr];
//                v2=vel[vaddrA];
//                w = 1.f/dt_d+v/dh_d;
////                w2= 1.f/dt_d+v2/dh_d;
 //               iaddr  =ix*block1+(ndab_d-1)*block2+(p_d-1);
 //               iaddrp2=ix*block1+(ndab_d-2)*block2+(p_d-1);

                //front face
//              awfld3[iaddr]=((awfld2[iaddrp2] - awfld3[iaddrp2] + awfld2[iaddr])/dt_d +\
//                                      v*(awfld2[iaddrp2] - awfld2[iaddr]   + awfld3[iaddrp2])/dh_d)/w;
                //back face
//                awfld3A[iaddr]=((awfld2A[iaddrp2] - awfld3A[iaddrp2] + awfld2A[iaddr])/dt_d +\
                                        v2*(awfld2A[iaddrp2] - awfld2A[iaddr]   + awfld3A[iaddrp2])/dh_d)/w2;
//        }

        // go down the "ladder"
//        if (ix < nx_d) {
//              vaddr=0*block3+ix;
//                vaddrA=(nz_d-1)*block3+ix;
//              vd  = (vel[vaddr]*vel[vaddr])*dtdh;
//                vd2 = (vel[vaddrA]*vel[vaddrA])*dtdh;
//                for(int ip=p_d-2; ip>= 0; ip--){
 //                       iaddr  =ix*block1+(ndab_d-1)*block2+ ip   ;
//                        iaddrp1=ix*block1+(ndab_d-1)*block2+(ip+1);
//                        iaddrp2=ix*block1+(ndab_d-2)*block2+(ip+1);
//                        iaddrd1=ix*block1+(ndab_d-1)*block2+ ip   ;
//                        iaddrd2=ix*block1+(ndab_d-2)*block2+ ip   ;

                        //front face
//                      awfld3[iaddr]=(1.f-vd)/(1.f+vd)*awfld2[iaddrd1]+\
//                                                (1.f+vd)/(1.f+vd)*awfld2[iaddrd2]+\
//                                               (-1.f+vd)/(1.f+vd)*awfld2[iaddrp2]+\
//                                               (-1.f-vd)/(1.f+vd)*awfld2[iaddrp1]+\
//                                               (-1.f+vd)/(1.f+vd)*awfld3[iaddrd2]+\
//                                                (1.f+vd)/(1.f+vd)*awfld3[iaddrp2]+\
//                                                (1.f-vd)/(1.f+vd)*awfld3[iaddrp1];
                        //bacl face
 //                       awfld3A[iaddr]=(1.f-vd2)/(1.f+vd2)*awfld2A[iaddrd1]+\
                                                   (1.f+vd2)/(1.f+vd2)*awfld2A[iaddrd2]+\
                                                  (-1.f+vd2)/(1.f+vd2)*awfld2A[iaddrp2]+\
                                                  (-1.f-vd2)/(1.f+vd2)*awfld2A[iaddrp1]+\
                                                  (-1.f+vd2)/(1.f+vd2)*awfld3A[iaddrd2]+\
                                           (1.f+vd2)/(1.f+vd2)*awfld3A[iaddrp2]+\
                                                   (1.f-vd2)/(1.f+vd2)*awfld3A[iaddrp1];
 //               }
 //       }


        //replace halo points with DAB points
  //      if (ix < nx_d ) {
                        //front face
//                      wfld3[0*block3+ix]=awfld3[ix*block1+4*block2];
//                      wfld3[1*block3+ix]=awfld3[ix*block1+3*block2];
//                      wfld3[2*block3+ix]=awfld3[ix*block1+2*block2];
//                      wfld3[3*block3+ix]=awfld3[ix*block1+1*block2];
                        //back face
//                        wfld3[(nz_d-1)*block3+ix]=awfld3A[ix*block1+4*block2];
//                        wfld3[(nz_d-2)*block3+ix]=awfld3A[ix*block1+3*block2];
//                        wfld3[(nz_d-3)*block3+ix]=awfld3A[ix*block1+2*block2];
//                        wfld3[(nz_d-4)*block3+ix]=awfld3A[ix*block1+1*block2];
//        }

//}
            
