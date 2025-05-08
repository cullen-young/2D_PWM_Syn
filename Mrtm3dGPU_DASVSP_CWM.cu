/* 2D RTM on GPU including offset gathers */

/*
  Copyright (C) 2012 University of Western Australia
  
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
#include <cuda.h>
#include <cuda_runtime_api.h>
extern "C" {
  #include <rsf.h>
}
#include "kernelwfld3dgpu_dasvsp_cwm.cu"

/* Checks the current GPU device for an error flag after the most recent 
 * CUDA command and prints to stderr */
static void sf_check_gpu_error (const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    sf_error ("Cuda error: %s: %s", msg, cudaGetErrorString (err));
}

/*------------------------------------------------------------*/
/* Main function */
/*------------------------------------------------------------*/
int main (int argc, char* argv[]) {

  int sdepth;       // . . depth in samples of source 
  int rdepth;       // . . depth in samples of receiver 
  int gpu;         // . . which gpu is used to do calculations (0 or 1)
  int nx,ny,nz,nt,nh,nr,nsx,nsy,np; // . . number of steps in each dimension
  int imgj;        // . . Imaging subsampling
  int wfldj;     // . . Wavefield subsampling
  int bwidth;  // . . Width of Random boundary
  int nsmooth; // . . Number of smoothing operations
  float dx,dy,dz,dt,ox,oy;  // . . sampling distance on each axis
  float var;       // . . density of velocity field
  int ictype;      // . . Imaging Condition type
  int itrevmin;      // . . Mininum time sample to reverse swf and rwf.
  int eps;
  bool verb;       // . . toggle verbose 
  bool wantwf;     // . . toggle full waveform output
  bool wantdat;    // . . Output data from source modelling
  bool wantillum;  // . . Output source illumination
    
  sf_file Fin ;    // . . input image file (with offsets)
  sf_file Fvel;    // . . input file is velocity, dimensions nx * nz
  sf_file Fswf;    // . . input SWF, dimension nx * nt
  sf_file Frwf;    // . . input RWF, dimension nx * nt
  sf_file Fout;    // . . output file, image, dimensions nh * nx * nz
  sf_file Fdat;    // . . output file, data,  dimensions nx * nt
  sf_file Fswfout; // . . output file, SWF at time zero, dimension 2 * nz * nx
  sf_file Frwfout; // . . output file, RWF at time zero, dimension 2 * nz * nx 
  sf_file Fillum;  // . . output file, illumination, dimension nz*nx
  
  sf_axis az, ax, ay, ah, at, ar, ap, atout, asx, asy;

  /*------------------------------------------------------------*/
  /* init RSF */
  sf_init(argc, argv);
  
  /* Read variables in from SConstruct */
  if (!sf_getbool("wantdat", &wantdat)) wantdat=false; /* BOOL: Extract data from SWF */
  if (!sf_getbool("wantwf", &wantwf)) wantwf=false; /* BOOL: Use full wavefield  */
  if (!sf_getbool("wantillum", &wantillum)) wantillum=false; /* BOOL: Write source illumination  */
  if (!sf_getbool("verb", &verb)) verb=true; /* BOOL: Be verbose */
  if (!sf_getint("imgj",&imgj)) imgj=1; /* INT: Number of subsample steps (for stabilitiy) */
  if (!sf_getint("wfldj",&wfldj)) wfldj=100; /* INT: Number of subsample steps (for stabilitiy) */
  if (!sf_getint("sdepth", &sdepth)) sdepth = 0 ; /*INT: Source depth */
  if (!sf_getint("rdepth", &rdepth)) rdepth = 0 ; /*INT: Receiver depth */
  if (!sf_getint("gpu", &gpu)) gpu = 0 ; /*INT: Which gpu to use */
  if (!sf_getfloat("var", &var)) var=0.; //FLOAT: Randomness steepness
  if (!sf_getint("bwidth",&bwidth)) bwidth=160; //Int: How far from the edge to randomize	
  if (!sf_getint("ictype",&ictype)) ictype=1; // Imaging Condition Type: 1=cross-correlation; 2=derivative
  if (!sf_getint("itrevmin",&itrevmin)) itrevmin=0; // Minimum time to reverse swf and rwf
  if (!sf_getint("nsmooth",&nsmooth)) nsmooth=1; // Smoothing of output image
  if (!sf_getint("eps",&eps)) eps=1e-5; // epsilon

  /*------------------------------------------------------------*/
  /* Commands to initialize GPU */
  cudaDeviceReset();//cudaThreadExit is deprecated.
  cudaSetDevice(gpu);
  
  sf_warning("using GPU #%d", gpu);
  cudaDeviceProp prop;
  int whichDevice;
  cudaGetDevice( &whichDevice ) ;
  cudaGetDeviceProperties( &prop, whichDevice );
  if (prop.canMapHostMemory != 1) {
    fprintf(stderr, "Device cannot map memory.\n" );
  }

  /* Input and output files */
  Fin  = sf_input("in");    // Input image size (nh,nx,ny,nz)
  Fvel = sf_input("vel");   // Velocity (nx,ny,nz) 
  Fswf = sf_input("swf");   // SWF (nr,nt,np)
  Frwf = sf_input("rwf");   // RWF (nsx,nsy,nt,np)
  Fout = sf_output("out");  // Image (nh,nx,ny,nz)
  Fdat = sf_output("data"); // Data (nr,nt)
  Fswfout = sf_output("swfout"); // Reversed SWF at zero time (nx,ny,nz,nw)
  Frwfout = sf_output("rwfout"); // Reversed RWF at zero time (nx,ny,nz,nw)
  Fillum = sf_output("illum"); // Reversed RWF at zero time (nx,ny,nz)
  
  /* Read in axes and labels */
  ax = sf_iaxa(Fvel,1); sf_setlabel(ax,"x"); if(verb) sf_raxa(ax); /* X */
  ay = sf_iaxa(Fvel,2); sf_setlabel(ay,"y"); if(verb) sf_raxa(ay); /* Y */
  az = sf_iaxa(Fvel,3); sf_setlabel(az,"z"); if(verb) sf_raxa(az); /* Z */
  ah = sf_iaxa(Fin, 1); sf_setlabel(ah,"h"); if(verb) sf_raxa(ah); /* H */

  // . . Downhole injection points in reciprocal sources
  ar = sf_iaxa(Fswf,1); sf_setlabel(ar,"r"); if(verb) sf_raxa(ar); /* R */
  at = sf_iaxa(Fswf,2); sf_setlabel(at,"t"); if(verb) sf_raxa(at); /* T */
  ap = sf_iaxa(Fswf,3); sf_setlabel(ap,"p"); if(verb) sf_raxa(ap); /* P */

  // . . Surface injection points in reciprocal receivers
  asx = sf_iaxa(Frwf,1); sf_setlabel(asx,"sx"); if(verb) sf_raxa(asx); /* SX */
  asy = sf_iaxa(Frwf,2); sf_setlabel(asy,"sy"); if(verb) sf_raxa(asy); /* SY */

  atout = sf_iaxa(Fswf,2); sf_setlabel(atout,"t"); 

  /* Read number of steps and sampling distance from labels */
  nx = sf_n(ax); dx = sf_d(ax); ox = sf_o(ax);
  ny = sf_n(ay); dy = sf_d(ay); oy = sf_o(ay);
  nz = sf_n(az); dz = sf_d(az); //oz = sf_o(az);
  nh = sf_n(ah); 	            //oh = sf_o(ah);
  nt = sf_n(at); dt = sf_d(at); //ot = sf_o(at);
  nr = sf_n(ar); //dr = sf_d(ar); //orr= sf_o(ar);
  np = sf_n(ap); //dp = sf_d(ap); //op = sf_o(ap);

  nsx = sf_n(asx); //dsx = sf_d(asx); //osx = sf_o(asx);
  nsy = sf_n(asy); //dsy = sf_d(asy); //osy = sf_o(asy);

  /* Set up time out axis */
  sf_setn(atout,1); sf_setd(atout,1.f); sf_seto(atout,0.f); 
   
  sf_oaxa(Fout,ah,1);
  sf_oaxa(Fout,ax,2);
  sf_oaxa(Fout,ay,3);
  sf_oaxa(Fout,az,4);   
   
  if (wantdat) {
    sf_oaxa(Fdat,ar,1);
    sf_oaxa(Fdat,at,2);
    sf_oaxa(Fdat,ap,3);
    sf_oaxa(Fdat,atout,4);
  }
  
  if (wantillum){
  	sf_oaxa(Fillum,ax,1);
	sf_oaxa(Fillum,ay,2);
    sf_oaxa(Fillum,az,3);
  }
   
  /* Set output file dimensions, nx * ny * nz * nt */
  sf_setn(atout,nt/(float)wfldj+1); sf_setd(atout,dt*wfldj); sf_seto(atout,0.); 
  if (wantwf) {
    sf_oaxa(Fswfout,ax,1);
    sf_oaxa(Fswfout,ay,2);
    sf_oaxa(Fswfout,az,3);
    sf_oaxa(Fswfout,atout,4);
    sf_oaxa(Fswfout,ap,5);
    
    sf_oaxa(Frwfout,ax,1);
    sf_oaxa(Frwfout,ay,2);
    sf_oaxa(Frwfout,az,3);
    sf_oaxa(Frwfout,atout,4);
    sf_oaxa(Frwfout,ap,5);
  }
 
  /***************************/
  /* . . VELOCITY DOMAIN . . */
  /***************************/
  /* . . velocity host - FULL VOLUME */
  float *vel_h=NULL;
  vel_h = sf_floatalloc(nx*ny*nz);
  sf_floatread(vel_h,nx*ny*nz,Fvel);
 
  /* . . velocity device - Keep same size as RWF and SWF*/
  float *vel_d;
  cudaMalloc((void**) &vel_d,nx*ny*nz*sizeof(float));
  cudaMemcpy(vel_d,vel_h,nx*ny*nz*sizeof(float),cudaMemcpyHostToDevice);

  /************************/
  /* . . IMAGE DOMAIN . . */
  /************************/
  /* . . Image host */
  float *img_h = NULL;
  img_h = sf_floatalloc(nx*ny*nz*nh);
  memset(img_h,0,nh*nx*ny*nz*sizeof(float));

  /* . . Image Device - Keep same size as rwf and swf */
  float *img_d;
  cudaMalloc((void**) &img_d,nx*ny*nz*nh*sizeof(float));
  cudaMemset(img_d,0,nh*nx*ny*nz*sizeof(float));
 
  /***************************/
  /* . .   DATA DOMAIN   . . */
  /***************************/
  /* . . Receiver Data host */
  float *rdat_h = NULL;
  rdat_h = sf_floatalloc(nx*ny);
  memset(rdat_h,0.,nx*ny*sizeof(float));
  
  /* . . Receiver Data device */
  float *rdat_d;
  cudaMalloc((void**) &rdat_d,nx*ny*sizeof(float));
  cudaMemset(rdat_d,0,        nx*ny*sizeof(float));
 
  /* . . Source Data host */
  float *sdat_h = NULL;
  sdat_h = sf_floatalloc(nr);
  memset(sdat_h,0.,nr*sizeof(float));
  
  /* . . Source Data device */
  float *sdat_d;
  cudaMalloc((void**) &sdat_d,nr*sizeof(float));
  cudaMemset(sdat_d,0,        nr*sizeof(float));

  /***************************/
  /* . .  WFLD DOMAIN    . . */
  /***************************/
  /* . . output source full waveform */
  float *wfldout_h = NULL;
  wfldout_h = sf_floatalloc(nz*nx*ny);
  memset(wfldout_h,0,nx*ny*nz*sizeof(float));

  /* . . SWF host */
  float *swf_h = NULL;
  swf_h = sf_floatalloc(nt*nr);
  memset(swf_h,0,nt*nr*sizeof(float));
  
  /* . . SWF device */
  float *swf_d;
  cudaMalloc((void**) &swf_d,nt*nr*sizeof(float));
  cudaMemset(swf_d,0,nt*nr*sizeof(float));

  /* . . RWF host */
  float *rwf_h = NULL;
  rwf_h = sf_floatalloc(nt*nsx*nsy);
  memset(rwf_h,0,nt*nsx*nsy*sizeof(float));
  
  /* . . RWF device */
  float *rwf_d;
  cudaMalloc((void**) &rwf_d,nt*nsx*nsy*sizeof(float));
  cudaMemset(rwf_d,0,nt*nsx*nsy*sizeof(float));
  
  /***************************/
  /* . .Illumination DOMAIN. . */
  /***************************/
  float *illum_d;
  if (wantillum){
  	cudaMalloc((void**) &illum_d,nx*ny*nz*sizeof(float));
  	cudaMemset(illum_d,0,nx*ny*nz*sizeof(float));
  }
  
  /* arrays that handle previous two steps in FD computation */
  /* source and receiver wavefields separate */
  float *swf_prev_d, *swf_next_d, *swf_curr_d, *rwf_prev_d, *rwf_next_d, *rwf_curr_d;
  cudaMalloc((void**) &swf_next_d, nx*ny*nz*sizeof(float));
  cudaMemset(swf_next_d, 0, nx*ny*nz*sizeof(float) );

  cudaMalloc((void**) &swf_curr_d, nx*ny*nz*sizeof(float));
  cudaMemset(swf_curr_d, 0, nx*ny*nz*sizeof(float) );

  cudaMalloc((void**) &swf_prev_d ,nx*ny*nz*sizeof(float));
  cudaMemset(swf_prev_d, 0, nx*ny*nz*sizeof(float) );
  
  cudaMalloc((void**) &rwf_next_d, nx*ny*nz*sizeof(float));
  cudaMemset(rwf_next_d, 0, nx*ny*nz*sizeof(float) );

  cudaMalloc((void**) &rwf_curr_d, nx*ny*nz*sizeof(float));
  cudaMemset(rwf_curr_d, 0, nx*ny*nz*sizeof(float) );

  cudaMalloc((void**) &rwf_prev_d, nx*ny*nz*sizeof(float));
  cudaMemset(rwf_prev_d, 0, nx*ny*nz*sizeof(float) );
  
  /* Reset time axis (for CFL stability) */
  int   nt_step = nt*imgj;
  float dt_step = dt/(float)(imgj);
  int   itrev_step=itrevmin*imgj;
  
  //putting symbols in constant memory which is more optimal...
  cudaMemcpyToSymbol(ny_d,   &ny, sizeof(int));
  cudaMemcpyToSymbol(nx_d,   &nx, sizeof(int));
  cudaMemcpyToSymbol(nz_d,   &nz, sizeof(int));
  cudaMemcpyToSymbol(nh_d,   &nh, sizeof(int));
  cudaMemcpyToSymbol(nt_d,   &nt, sizeof(int));
  cudaMemcpyToSymbol(nr_d,   &nr, sizeof(int));
  cudaMemcpyToSymbol(np_d,   &np, sizeof(int));
  cudaMemcpyToSymbol(nsx_d,  &nsx,sizeof(int));
  cudaMemcpyToSymbol(nsy_d,  &nsy,sizeof(int));
  cudaMemcpyToSymbol(wfld_j_d,&wfldj, sizeof(int));
  cudaMemcpyToSymbol(dt_d,  &dt_step, sizeof(float));
  cudaMemcpyToSymbol(dx_d,  &dx, sizeof(float));
  cudaMemcpyToSymbol(dy_d,  &dy, sizeof(float));
  cudaMemcpyToSymbol(dz_d,  &dz, sizeof(float));
  cudaMemcpyToSymbol(eps_d, &eps,sizeof(float));

  /* CUDA BLOCK STRUCTURES */
  dim3 dimBlock2Ds,dimGrid2Ds,dimBlock2D,dimGrid2D,dimBlock1D,dimGrid1D;

sf_warning("nr,nx,ny,nz %d %d %d %d",nr,nx,ny,nz);

  /* For 1D kernels */
  dimBlock1D = dim3(256,1);
  dimGrid1D  = dim3(ceil(nr/256.f),1);

  /* For 2D kernels */
  dimBlock2D = dim3(16,16);
  dimGrid2D  = dim3(ceil(nx/16.f),ceil(ny/16.f));

  /* For 2D kernels */
  dimBlock2Ds = dim3(16,16);
  dimGrid2Ds  = dim3(ceil(nx/16.f),ceil(ny/16.f));

  // . . Define FD coefficients (from Define in wave_kernel)
  float dh = sqrt( dx*dx + dy*dy + dz*dz ) / sqrt(3.f) ;
  sf_warning("dh %g",dh);
  
  float cc_h[COEFFS_SIZE]; 
  cc_h[0]=-205.f/72.f *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[1]=   8.f/5.f  *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[2]=  -1.f/5.f  *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[3]=   8.f/315.f*(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[4]=  -1.f/560.f*(dt_step)*(dt_step)/(dh)/(dh);
  cudaMemcpyToSymbol(cc_d, cc_h, COEFFS_SIZE*sizeof(float));

  /* Extra array for rotation */
  sf_warning("Starting loop over sources");

  // . . Initialize image space
  cudaMemset( img_d, 0, nh*nx*ny*nz*sizeof(float) );
  
  // Define injection location surface coordinate
  int isx = (int)((2.f-ox)/dx);
  int isy = (int)((2.f-oy)/dy);
  sf_warning("isx: %d", isx);
  sf_warning("isy: %d", isy);
  
  /* Loop over conical waves */
  for (int ip=0; ip<np; ++ip) {

	/* . . Read in data . . */
	sf_floatread(swf_h,    nt*nr,Fswf);
	cudaMemcpy(swf_d,swf_h,nt*nr*sizeof(float), cudaMemcpyHostToDevice);
	
	sf_floatread(rwf_h,    nt*nsx*nsy,Frwf);
	cudaMemcpy(rwf_d,rwf_h,nt*nsx*nsy*sizeof(float), cudaMemcpyHostToDevice);

	/* . . Set propagation wavefields and temp image to zero . . */
	cudaMemset(swf_prev_d, 0,nx*ny*nz*sizeof(float) );
	cudaMemset(swf_next_d, 0,nx*ny*nz*sizeof(float) );
	cudaMemset(swf_curr_d, 0,nx*ny*nz*sizeof(float) );
	
	cudaMemset(rwf_prev_d, 0,nx*ny*nz*sizeof(float) );
	cudaMemset(rwf_next_d, 0,nx*ny*nz*sizeof(float) );
	cudaMemset(rwf_curr_d, 0,nx*ny*nz*sizeof(float) );

    /* Randomize the boundaries */
	//if (bwidth > 0 && var > 0.00000001) {
 	//	randomize_kernel<<<dimGrid2D,dimBlock2D>>>(vel_d,var,bwidth,nx,nz);
    	//}
    
	/* Source wavefield propagated all the way through the model first */
	for (int it=0; it<nt_step; ++it){
  
		int itnew = (float)(it)/(float)(imgj);

		if(verb && itnew%10==0) fprintf(stderr,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bSWF SHOT ip=%d t=%d  ",ip,itnew);
	
		/* Inject energy into SWF */
		if (it < nt_step-2*imgj)  {
			inject_linearborehole_3d<<<dimGrid1D,dimBlock1D>>>(isx,isy,itnew,swf_d,swf_curr_d,imgj);
	  		sf_check_gpu_error ("INJECT WF (1)");  
	  	}

	  	/* Extract data at current time step */
	  	if (wantdat && it%imgj==0) {
	  		extract_kernel3d<<<dimGrid1D,dimBlock1D>>>(isx,isy,swf_curr_d,sdat_d);				
	  		cudaMemcpy(sdat_h,sdat_d,nr*sizeof(float),cudaMemcpyDeviceToHost);
	  		sf_floatwrite(sdat_h,nr,Fdat);
	  		sf_check_gpu_error ("EXTRACT DATA");
	  	}
      	
      	if(wantillum && it%imgj==0){
      		source_illum3d<<<dimGrid2D,dimBlock2D>>>(illum_d,swf_curr_d);
      		sf_check_gpu_error ("ILLUMI");
      	}

  	  	// . . Source Wavefield check
		if (wantwf && it%(imgj*wfldj)==0) {  
     		cudaMemcpy(wfldout_h, swf_curr_d, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
      		sf_floatwrite(wfldout_h, nz*ny*nx, Fswfout);
 	  	}  
  
	  	/* Actual finite-difference algorithm is calculated here */
	  	propagate_step3d<<<dimGrid2Ds,dimBlock2Ds>>>(swf_prev_d,swf_curr_d,swf_next_d,vel_d);
		sf_check_gpu_error ("PROPAGATE SWF (1)");


  	  	/* Rotate wavefields and iterate the timestep forwards by one */
	  	swf_prev_d = swf_curr_d; 
	  	swf_curr_d = swf_next_d; 
	  	swf_next_d = swf_prev_d;
  	  	sf_check_gpu_error ("ROTATE WFLDS");

	} /* END FORWARD TIME PROPAGATION LOOP */
  
  	/* Rotate source wavefields for back-propagation */
  	swf_prev_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = swf_prev_d;

  	/* . . Propagate both wavefields backwards through model and apply IC */
  	for(int it=nt_step-1; it>=itrev_step; --it){
  	 
  	 	int itnew = (float)(it)/(float)(imgj);

  		//int itnew = (float)(it)/(float)(imgj);  	  
	  	if(verb && itnew%10==0) fprintf(stderr,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bRWF SHOT ip=%d t=%d  ",ip,itnew);

	  	/* INJECT Receiver data */	
	   	if (it < nt_step-imgj) {
	   		inject_kernel_receiver_3d<<<dimGrid2D,dimBlock2D>>>(rdepth,itnew,rwf_d,rwf_curr_d,imgj);
			sf_check_gpu_error ("INJECT KERNEL (2)");   
		}   	 
	   	
		if (wantwf && it%(imgj*wfldj)==0) {  
	    
	   		/* RWF output */
	   		cudaMemcpy(wfldout_h, rwf_curr_d, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
	   		sf_floatwrite(wfldout_h, nz*ny*nx, Frwfout);  
 	  	} 

  	  	//if (it%imgj==0) {
	  		/* IMAGING */
		//	if (ictype == 1) {
		  		//image_kernel<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
		//		image_kernel<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
		//	} else if (ictype==2) {
		//  		derivative_xig_kernel<<<dimGridIM,dimBlockIM>>>(vel_d,swf_curr_d,rwf_curr_d,swf_next_d,rwf_next_d,img_d);
		//		sf_check_gpu_error ("XIG KERNEL (2)");
		//	} else if (ictype==3) {
		//		image_decon<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
		//	}
  	  	//}
	
	    /* IMAGING */
		if (it%imgj==0) {
            image_xcorr_3d<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
        	sf_check_gpu_error ("INJECT WF (1)");	
		}
  
  	  	/* PROPAGATE SWF */	
  	  	propagate_step3d<<<dimGrid2Ds,dimBlock2Ds>>>(swf_prev_d,swf_curr_d,swf_next_d,vel_d);
		sf_check_gpu_error ("PROPAGATE SWF (2)");
	
  	  	/* PROPAGATE RWF  */	
  	  	propagate_step3d<<<dimGrid2Ds,dimBlock2Ds>>>(rwf_prev_d,rwf_curr_d,rwf_next_d,vel_d);
		sf_check_gpu_error ("PROPAGATE RWF (2)"); 
	   
		/* ROTATE WAVEFIELDS */
        swf_prev_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = swf_prev_d;
	  	rwf_prev_d = rwf_curr_d; rwf_curr_d = rwf_next_d; rwf_next_d = rwf_prev_d;

	} /* . . END TIME LOOP */
  

  } /* . . End shot loop . . */
 
  /* Extract image_d to img_h, which can be written onto the local disk */
  cudaMemcpy(img_h, img_d, nx*ny*nz*nh*sizeof(float), cudaMemcpyDeviceToHost);

  sf_floatwrite(img_h, nz*ny*nx*nh, Fout);
  sf_warning("\n"); 
  
  if(wantillum){
  	cudaMemcpy(img_h, illum_d, nx*ny*nz*sizeof(float), cudaMemcpyDeviceToHost);
  	sf_floatwrite(img_h, nz*ny*nx, Fillum);
  }
 
  /* Cleaning up the memory */
  cudaFree(vel_d);  cudaFree(swf_d);  cudaFree(rwf_d);  cudaFree(img_d); 
  cudaFree(swf_curr_d);  cudaFree(swf_next_d); cudaFree(swf_prev_d);
  cudaFree(rwf_curr_d);  cudaFree(rwf_next_d); cudaFree(rwf_prev_d);
  free(vel_h); free(swf_h); free(rwf_h); free(img_h); free(wfldout_h);

  exit(0);
}
