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
  int nx,nz,nt,nh,nr,nsx,np; // . . number of steps in each dimension
  int imgj;        // . . Imaging subsampling
  int wfldj;     // . . Wavefield subsampling
  int bwidth;  // . . Width of Random boundary
  int nsmooth; // . . Number of smoothing operations
  float dx,dz,dt,dr,dsx,dp,op,oh,ot,orr,osx,ory,os,oix,dix,ox,oz;  // . . sampling distance on each axis
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
  
  sf_axis az, ax, ah, at, ar, ap,  atout, asx, aix;

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
  if (!sf_getint("ictype",&ictype)) ictype=2; // Imaging Condition Type: 1=cross-correlation; 2=derivative
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
  Fin  = sf_input("in");    // Input image size (nh,nx,nz)
  Fvel = sf_input("vel");   // Velocity (nx,nz) 
  Fswf = sf_input("swf");   // SWF (nr,nt,np)
  Frwf = sf_input("rwf");   // RWF (nsx,nt,np)
  Fout = sf_output("out");  // Image  (nh,nx,nz)
  Fdat = sf_output("data"); // Data (nx,nt)
  Fswfout = sf_output("swfout"); // Reversed SWF at zero time (nx,nh,2)
  Frwfout = sf_output("rwfout"); // Reversed RWF at zero time (nx,nh,2)
  Fillum = sf_output("illum"); // Reversed RWF at zero time (nx,nh,2)
  
  /* Read in axes and labels */
  az = sf_iaxa(Fvel,2); sf_setlabel(az,"z"); if(verb) sf_raxa(az); /* Z */
  ax = sf_iaxa(Fvel,1); sf_setlabel(ax,"x"); if(verb) sf_raxa(ax); /* X */
  ah = sf_iaxa(Fin, 1); sf_setlabel(ah,"h"); if(verb) sf_raxa(ah); /* H */

  at = sf_iaxa(Fswf,2); sf_setlabel(at,"t"); if(verb) sf_raxa(at); /* T */
  ar = sf_iaxa(Fswf,1); sf_setlabel(ar,"r"); if(verb) sf_raxa(ar); /* R */
  ap = sf_iaxa(Fswf,3); sf_setlabel(ap,"p"); if(verb) sf_raxa(ap); /* P */

  asx = sf_iaxa(Frwf,1); sf_setlabel(asx,"sx"); if(verb) sf_raxa(asx); /* SX */
  //ary = sf_iaxa(Frwf,3); sf_setlabel(ary,"ry"); if(verb) sf_raxa(ary); /* RY */

  //aix= sf_iaxa(Fin,2); sf_setlabel(as,"ix"); if(verb) sf_raxa(aix); /* Image x */
  atout = sf_iaxa(Fswf,2); sf_setlabel(atout,"t"); 

  /* Read number of steps and sampling distance from labels */
  nx = sf_n(ax); dx = sf_d(ax); ox = sf_o(ax);
  nz = sf_n(az); dz = sf_d(az); oz = sf_o(az);
  nh = sf_n(ah); 	        oh = sf_o(ah);
  nt = sf_n(at); dt = sf_d(at); ot = sf_o(at);
  nr = sf_n(ar); dr = sf_d(ar); orr= sf_o(ar);
  np = sf_n(ap); dp = sf_d(ap); op = sf_o(ap);

  nsx = sf_n(asx); dsx = sf_d(asx); osx = sf_o(asx);
  //nry = sf_n(ary); dry = sf_d(ary); ory = sf_o(ary);
  
  //ns = sf_n(as); ds = sf_d(as); os = sf_o(as);
  //nix= sf_n(aix); dix=sf_d(aix); oix=sf_o(aix); 
 
  /* Set up time out axis */
  sf_setn(atout,1); sf_setd(atout,1.f); sf_seto(atout,0.f); 
   
  sf_oaxa(Fout,ah,1);
  sf_oaxa(Fout,ax,2);
  sf_oaxa(Fout,az,3);   
   
  if (wantdat) {
    sf_oaxa(Fdat,ar,1);
    sf_oaxa(Fdat,at,2);
    sf_oaxa(Fdat,ap,3);
  }
  
  if (wantillum){
  	sf_oaxa(Fillum,ax,1);
    	sf_oaxa(Fillum,az,2);
  }
   
  /* Set output file dimensions, nx * nz * nt */
  sf_setn(atout,nt/(float)wfldj+1); sf_setd(atout,dt*wfldj); sf_seto(atout,0.); 
  if (wantwf) {
    sf_oaxa(Fswfout,ax,1);
    sf_oaxa(Fswfout,az,2);
    sf_oaxa(Fswfout,atout,3);
    sf_oaxa(Fswfout,ap,4);
    
    sf_oaxa(Frwfout,ax,1);
    sf_oaxa(Frwfout,az,2);
    sf_oaxa(Frwfout,atout,3);
    sf_oaxa(Frwfout,ap,4);
  }
 
  /***************************/
  /* . . VELOCITY DOMAIN . . */
  /***************************/
  /* . . velocity host - FULL VOLUME */
  float *vel_h=NULL;
  vel_h = sf_floatalloc(nx*nz);
  sf_floatread(vel_h,nx*nz,Fvel);
  /* . . Cut vel on host */
  //float *velcut_h=NULL;
  //velcut_h = sf_floatalloc(nx*nz);
  //memset(velcut_h,0,nx*nz*sizeof(float));
  /* . . velocity device - Keep same size as RWF and SWF*/
  float *vel_d;
  cudaMalloc((void**) &vel_d,nx*nz*sizeof(float));
  cudaMemset(vel_d,0,nx*nz*sizeof(float));

  /************************/
  /* . . IMAGE DOMAIN . . */
  /************************/
  /* . . Image host */
  float *img_h = NULL;
  img_h = sf_floatalloc(nx*nz*nh);
  memset(img_h,0,nh*nx*nz*sizeof(float));
  /* . . Cut Image host */
  //float *imgcut_h = NULL;
  //imgcut_h = sf_floatalloc(nx*nz*nh);
  //memset(imgcut_h,0,nh*nx*nz*sizeof(float));
  /* . . Image Device - Keep same size as rwf and swf */
  float *img_d;
  cudaMalloc((void**) &img_d,nx*nz*nh*sizeof(float));
  cudaMemset(img_d,0,nh*nx*nz*sizeof(float));
  //float *imgtmp_d;
  //cudaMalloc((void**) &imgtmp_d,nx*nz*nh*sizeof(float));
  //cudaMemset(imgtmp_d,0,nh*nx*nz*sizeof(float));

  /***************************/
  /* . .   DATA DOMAIN   . . */
  /***************************/
  /* . . Data host */
  float *rdat_h = NULL;
  rdat_h = sf_floatalloc(nx);
  memset(rdat_h,0.,nx*sizeof(float));
  /* . . Data device */
  float *rdat_d;
  cudaMalloc((void**) &rdat_d,nx*sizeof(float));
  cudaMemset(rdat_d,0,        nx*sizeof(float));
 

  float *sdat_h = NULL;
  sdat_h = sf_floatalloc(nr);
  memset(sdat_h,0.,nr*sizeof(float));
  /* . . Data device */
  float *sdat_d;
  cudaMalloc((void**) &sdat_d,nr*sizeof(float));
  cudaMemset(sdat_d,0,        nr*sizeof(float));

  /***************************/
  /* . .  WFLD DOMAIN    . . */
  /***************************/
  /* . . output source full waveform */
  float *wfldout_h = NULL;
  wfldout_h = sf_floatalloc(nz*nx);
  memset(wfldout_h,0,nx*nz*sizeof(float));

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
  rwf_h = sf_floatalloc(nt*nsx);
  memset(rwf_h,0,nt*nsx*sizeof(float));
  /* . . RWF device */
  float *rwf_d;
  cudaMalloc((void**) &rwf_d,nt*nsx*sizeof(float));
  cudaMemset(rwf_d,0,nt*nsx*sizeof(float));
  
  /***************************/
  /* . .Illumination DOMAIN. . */
  /***************************/
  float *illum_d;
  if (wantillum){
  	cudaMalloc((void**) &illum_d,nx*nz*sizeof(float));
  	cudaMemset(illum_d,0,nx*nz*sizeof(float));
  }
  
  /* arrays that handle previous two steps in FD computation */
  /* source and receiver wavefields separate */
  float *swf_next_d, *swf_curr_d, *swf_tmp_d, *rwf_next_d, *rwf_curr_d;
  cudaMalloc((void**) &swf_next_d, nx*nz*sizeof(float) );
  cudaMalloc((void**) &swf_curr_d, nx*nz*sizeof(float) );
  cudaMalloc((void**) &swf_tmp_d , nx*nz*sizeof(float) );
  cudaMalloc((void**) &rwf_next_d, nx*nz*sizeof(float) );
  cudaMalloc((void**) &rwf_curr_d, nx*nz*sizeof(float) );
  cudaMemset(swf_next_d, 0., nx*nz*sizeof(float) );
  cudaMemset(swf_curr_d, 0., nx*nz*sizeof(float) );
  cudaMemset(rwf_next_d, 0., nx*nz*sizeof(float) );
  cudaMemset(rwf_curr_d, 0., nx*nz*sizeof(float) );
  
  /* Reset time axis */
  int   nt_step = nt*imgj;
  float dt_step = dt/(float)imgj;
  int   itrev_step=itrevmin*imgj;
  
  //putting symbols in constant memory which is more optimal...
  cudaMemcpyToSymbol(nz_d,   &nz, sizeof(int));
  cudaMemcpyToSymbol(nx_d,   &nx, sizeof(int));
  cudaMemcpyToSymbol(nh_d,   &nh, sizeof(int));
  cudaMemcpyToSymbol(nt_d,   &nt, sizeof(int));
  cudaMemcpyToSymbol(nr_d,   &nr, sizeof(int));
  cudaMemcpyToSymbol(np_d,   &np, sizeof(int));
  cudaMemcpyToSymbol(nsx_d,  &nsx, sizeof(int));
  cudaMemcpyToSymbol(wfld_j_d,&wfldj, sizeof(int));
  cudaMemcpyToSymbol(dt_d,  &dt_step, sizeof(float));
  cudaMemcpyToSymbol(dx_d,  &dx, sizeof(float));
  cudaMemcpyToSymbol(dz_d,  &dz, sizeof(float));
  cudaMemcpyToSymbol(eps_d,   &eps, sizeof(int));


  /* CUDA BLOCK STRUCTURES */
  dim3 dimBlock1D,dimGrid1D,dimBlock2D,dimGrid2D,dimBlockIM,dimGridIM,dimBlock1Dr,dimGrid1Dr,dimBlock1Ds,dimGrid1Ds;

  /* For 1D kernels */
  dimBlock1D = dim3(256,1);
  dimGrid1D  = dim3(ceil(nx/256.f),1);

  /* For 2D kernels */
  dimBlock2D = dim3(16,16);
  dimGrid2D  = dim3(ceil(nx/16.f),ceil(nz/16.f));

  /* For imaging kernels */
  dimBlockIM = dim3(256,1);
  dimGridIM  = dim3(ceil(nx/256.f),nz);

  /* For 1D kernels */
  dimBlock1Dr = dim3(256,1);
  dimGrid1Dr  = dim3(ceil(nr/256.f),1);

  dimBlock1Ds = dim3(256,1);
  dimGrid1Ds = dim3(ceil(nsx/256.f),1);

  // . . Define coefficients (from Define in wave_kernel)
  float dh = sqrt( dx*dx + dz*dz ) / sqrt(2.f) ;
  float cc_h[COEFFS_SIZE]; 
  cc_h[0]=-205.f/72.f *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[1]=   8.f/5.f  *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[2]=  -1.f/5.f  *(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[3]=   8.f/315.f*(dt_step)*(dt_step)/(dh)/(dh);
  cc_h[4]=  -1.f/560.f*(dt_step)*(dt_step)/(dh)/(dh);
  cudaMemcpyToSymbol(cc_d, cc_h, COEFFS_SIZE*sizeof(float));

  /* Extra array for rotation */
  float *wtmp_d;
  cudaMalloc((void**) &wtmp_d,  nx*nz*sizeof(float));
  cudaMemset(          wtmp_d,0,nx*nz*sizeof(float));

  sf_warning("Starting loop over sources");

  cudaMemset(     img_d, 0., nh*nx*nz*sizeof(float) );
  
  /* Loop over time delays */
  for (int ip=0; ip<np; ++ip) {

	/* . . Read in data . . */
	sf_floatread(swf_h,    nt*nr,Fswf);
	cudaMemcpy(swf_d,swf_h,nt*nr*sizeof(float), cudaMemcpyHostToDevice);
	sf_floatread(rwf_h,    nt*nsx,Frwf);
	cudaMemcpy(rwf_d,rwf_h,nt*nsx*sizeof(float), cudaMemcpyHostToDevice);

	/* . . Set propagation wavefields and temp image to zero . . */
	cudaMemset(swf_next_d, 0.,    nx*nz*sizeof(float) );
	cudaMemset(swf_curr_d, 0.,    nx*nz*sizeof(float) );
	cudaMemset(rwf_next_d, 0.,    nx*nz*sizeof(float) );
	cudaMemset(rwf_curr_d, 0.,    nx*nz*sizeof(float) );
	cudaMemset(    wtmp_d, 0.,    nx*nz*sizeof(float) );

	/* . . Location where to insert subimage . . */
	/*int isloc = abs((int)(is*ds+os+ox-oix)/dix);  
	sf_warning("is ds os ox oix dix: %d %g %g %g %g %g",is,ds,os,ox,oix,dix);
    sf_warning("Reading in source %d with image location at %d",is,isloc);
    	*/

    	int isx = (int)((2.0-ox)/dx);
     	sf_warning("isx: %d", isx);

	/* Pass cut velocity model on to GPU */
 	cudaMemcpy(vel_d,vel_h,nx*nz*sizeof(float),cudaMemcpyHostToDevice);

    /* Randomize the boundaries */
   	if (bwidth > 0 && var > 0.00000001) {
 		randomize_kernel<<<dimGrid2D,dimBlock2D>>>(vel_d,var,bwidth,nx,nz);
    	}
    
	/* Source wavefield propagated all the way through the model first */
	for (int it=0; it<nt_step; ++it){
  
		int itnew = (float)(it)/(float)(imgj);

		if(verb && it%100==0) fprintf(stderr,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bSWF SHOT ip=%d t=%d  ",ip,itnew);
	
		/* Inject energy into SWF */
	  	if (itnew < nt-2)  inject_linearborehole2d<<<dimGrid1Dr,dimBlock1Dr>>>(isx,it,swf_d,swf_curr_d,imgj);
	  	sf_check_gpu_error ("INJECT WF (1)");  

  	  	if (it%imgj==0) {

			/* Extract data at current time step */
			if (wantdat) {
				extract_kernel<<<dimGrid1Dr,dimBlock1Dr>>>(isx,swf_curr_d,sdat_d);
				sf_check_gpu_error ("EXTRACT DATA");
				cudaMemcpy(sdat_h,sdat_d,nr*sizeof(float),cudaMemcpyDeviceToHost);
				sf_floatwrite(sdat_h,nr,Fdat);
			}
      	}
      	
      	if(wantillum){
      		source_illum<<<dimGrid2D,dimBlock2D>>>(illum_d,swf_curr_d);
      		sf_check_gpu_error ("ILLUMI");
      	}
          
	  	/* Actual finite-difference algorithm is calculated here */
	  	propagate_kernel<<<dimGrid2D,dimBlock2D>>>(swf_next_d,swf_curr_d,vel_d);
	  	sf_check_gpu_error ("PROPAGATE SWF (1)");
  	  
  	  	/* Rotate wavefields and iterate the timestep forwards by one */
	  	wtmp_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = wtmp_d;
  	  
	} /* END FORWARD TIME PROPAGATION LOOP */
  
  	/* Rotate source wavefields for back-propagation */
  	wtmp_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = wtmp_d;

  	/* Now propagate both wavefields backwards through model and correlate */
  	for(int it=nt_step-1; it>=itrev_step; --it){
  	 
  		int itnew = (float)(it)/(float)(imgj);  	  
	  	if(verb && it%100==0) fprintf(stderr,"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bRWF SHOT ip=%d t=%d  ",ip,itnew);

	  	/* INJECT Receiver data */	
	   	if (itnew > 1) inject_kernel_reverse_new<<<dimGrid1Ds,dimBlock1Ds>>>(rdepth,it,rwf_d,rwf_curr_d,imgj);
	   	sf_check_gpu_error ("INJECT KERNEL (2)");      	 

		//if (itnew < nt-2) inject_linearborehole2d_inverse<<<dimGrid1Ds,dimBlock1Ds>>>(isx,it,swf_d,swf_curr_d,imgj);
	   	
		if (wantwf && it%(imgj*wfldj)==0) {  

 	    		/* SWF output */
 	    		cudaMemcpy(wfldout_h, swf_curr_d, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
 	    		sf_floatwrite(wfldout_h, nz*nx, Fswfout);
 	    
 	    		/* RWF output */
 	    		cudaMemcpy(wfldout_h, rwf_curr_d, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
 	    		sf_floatwrite(wfldout_h, nz*nx, Frwfout);  
 	  	} 

  	  	//if (it%imgj==0) {
	  		/* IMAGING */
		//	if (ictype == 1) {
		  //		image_kernel<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
		//	} else if (ictype==2) {
		  //		derivative_xig_kernel<<<dimGridIM,dimBlockIM>>>(vel_d,swf_curr_d,rwf_curr_d,swf_next_d,rwf_next_d,img_d);
		//		sf_check_gpu_error ("XIG KERNEL (2)");
		//	} else if (ictype==3) {
		//		image_decon<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
		//	}
  	  	//}
	
  	  	/* PROPAGATE SWF */	
  	  	propagate_kernel<<<dimGrid2D,dimBlock2D>>>(swf_next_d,swf_curr_d,vel_d);
  	  	sf_check_gpu_error ("PROPAGATE SWF (2)");
	    
		if (itnew < nt-2) inject_linearborehole2d_inverse<<<dimGrid1Ds,dimBlock1Ds>>>(isx,it,swf_d,swf_curr_d,imgj);
	    
	    
  	  	/* ROTATE WAVEFIELDS */
	  	//wtmp_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = wtmp_d;
    
  	  	/* PROPAGATE RWF  */	
  	  	propagate_kernel<<<dimGrid2D,dimBlock2D>>>(rwf_next_d,rwf_curr_d,vel_d); 
  	  	sf_check_gpu_error ("PROPAGATE RWF (2)"); 
	   
		if (it%imgj==0) {
                        /* IMAGING */
                        if (ictype == 1) {
                                image_kernel<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
                        } else if (ictype==2) {
                                derivative_xig_kernel<<<dimGridIM,dimBlockIM>>>(vel_d,swf_curr_d,rwf_curr_d,swf_next_d,rwf_next_d,img_d);
                                sf_check_gpu_error ("XIG KERNEL (2)");
                        } else if (ictype==3) {
                                image_decon<<<dimGrid2D,dimBlock2D>>>(swf_curr_d,rwf_curr_d,img_d);
                        }
                }

		/* ROTATE WAVEFIELDS */
                wtmp_d = swf_curr_d; swf_curr_d = swf_next_d; swf_next_d = wtmp_d;

        //sf_check_gpu_error ("INJECT WF (1)");	
  	  	/* ROTATE WAVEFIELDS */
	  	wtmp_d = rwf_curr_d; rwf_curr_d = rwf_next_d; rwf_next_d = wtmp_d;

	} /* . . END TIME LOOP */
  

  } /* . . End shot loop . . */
 
  /* Extract image_d to img_h, which can be written onto the local disk */
  cudaMemcpy(img_h, img_d, nx*nz*nh*sizeof(float), cudaMemcpyDeviceToHost);

  sf_floatwrite(img_h, nz*nx*nh, Fout);
  sf_warning("\n");
  
  if(wantillum){
  	cudaMemcpy(img_h, illum_d, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
  	sf_floatwrite(img_h, nz*nx, Fillum);
  }
 
  /* Cleaning up the memory */
  cudaFree(vel_d);  cudaFree(swf_d);  cudaFree(rwf_d);  cudaFree(img_d); 
  cudaFree(swf_curr_d);  cudaFree(swf_next_d);
  cudaFree(swf_tmp_d);
  cudaFree(rwf_curr_d);  cudaFree(rwf_next_d);
  free(vel_h); free(swf_h); free(rwf_h); //free(velcut_h); free(imgcut_h);
  free(img_h); free(wfldout_h);

  exit(0);
}
