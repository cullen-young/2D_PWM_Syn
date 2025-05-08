#include <math.h>
#include <rsf.h>

// OPENMP support
#ifdef _OPENMP
#include <omp.h>
#endif

void shot2plane1p(sf_complex **out, sf_complex ***in, float **vel, float px, 
		  float ow, float dw, int nw,
		  float or, float dr, int nr, 
		  float os, float ds, int ns);
		  
void zeroout(sf_complex **out, int nw, int nr);

/*------------------------------------------------------------*/
int main(int argc, char* argv[])
{
	/* Variables */
	int np, nw, nr, ns, nx, nz;

	float op, dp, px;
	float ow, dw;
	float or, dr;
	float os, ds;

	float **vel;

	/* I/O files */
	sf_file Fin =NULL;
	sf_file Fvel=NULL;
	sf_file Fout=NULL;
	
	sf_axis aw, ar, as, ap, ax, az;
	/*----------------------------------------------------*/
	sf_init(argc,argv);

    /*------------------------------------------------------------*/
    /* OMP parameters */
#ifdef _OPENMP
    omp_init();
#endif
    /*------------------------------------------------------------*/


	if(! sf_getfloat("op", &op)) op=0.0;
	if(! sf_getfloat("dp", &dp)) dp=0.001;

	if(! sf_getint("np",   &np)) np=1;

	/*---------------------------------------------------*/
	Fin  = sf_input ("in" ); sf_settype(Fin,SF_COMPLEX);
	Fvel = sf_input ("vel");
	Fout = sf_output("out"); sf_settype(Fout,SF_COMPLEX);

	aw = sf_iaxa(Fin,1);
	ar = sf_iaxa(Fin,2);
	as = sf_iaxa(Fin,3);
	ax = sf_iaxa(Fvel,2);
	az = sf_iaxa(Fvel,1);
	ap = sf_maxa(np,op,dp);

	nw = sf_n(aw); dw = sf_d(aw); ow = sf_o(aw);
	nr = sf_n(ar); dr = sf_d(ar); or = sf_o(ar);
	ns = sf_n(as); ds = sf_d(as); os = sf_o(as);
	nx = sf_n(ax);
	nz = sf_n(az);
	
	sf_oaxa(Fout,aw,1);
	sf_oaxa(Fout,ar,2);
	sf_oaxa(Fout,ap,3);

	sf_complex ***in;
	in  = sf_complexalloc3(nw,nr,ns);
	vel = sf_floatalloc2(nz,nx);
	sf_complex **out;
	out = sf_complexalloc2(nw,nr);
	sf_complexread(in[0][0],nw*nr*ns,Fin);
	sf_floatread(vel[0],nz*nx,Fvel);

	/*---------------------------------------------------*/
	/*Main Loop*/

	/*All Shots*/
	/*Phase Encode*/
	for (int ip=0; ip<np; ip++){
		px = (float)(ip)*dp + op;

		sf_warning("NOW ON IP: %d",ip);
		
		zeroout(out, nw, nr);

		shot2plane1p(out, in, vel, px,
           			 ow, dw, nw,
				 or, dr, nr,
	                 	 os, ds, ns);

		sf_complexwrite(out[0], nw*nr, Fout);

//		zeroout(out, nw, nr);
	}

	free(**in);  free(*in);  free(in);
	             free(*out); free(out); 
	exit(0);
}

void shot2plane1p(sf_complex **out, sf_complex ***in, float **vel, float px,
                  float ow, float dw, int nw,
		 		  float or, float dr, int nr,
                  float os, float ds, int ns)
{
	int is,iw,ir;
	float xdt = ds*px;
	float ww,tx,vx;

	tx = 0.;

	for (is=0; is<ns; is++){
		//float sx = (is-1.)*ds+os;
		//tx = (float)(is)*xdt;
		tx += xdt*(vel[400][0]/vel[400][is]);
		//tx += xdt;
		//vx = (float) vel[400][is]/vel[400][0];
		//sf_warning("vx: %g",vx);
#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic,10)		\
    private(ir,iw,ww)				\
    shared(xdt,tx,vx,out,in,dw,ow,nw,nr)
#endif
		for (ir=0; ir<nr; ir++){
            //float rx=(ir-1.)*dr+or;
         	//int ix = SF_MIN(SF_MAX((floor)((sx+rx-or)/dr),0),nr-1);
			for (iw=0; iw<nw; iw++){
				ww = (float)(iw)*dw+ow;
				out[ir][iw] += in[is][ir][iw]*sf_cmplx(cosf(2.*SF_PI*ww*tx),sinf(2.*SF_PI*ww*tx));
			}
		}
	}
}

void zeroout(sf_complex **out, int nw, int nr)
{

	int ir,iw;

#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic)		\
    private(ir,iw)				\
    shared(out,nw,nr)
#endif
	for (ir=0; ir<nr; ir++){
		for (iw=0; iw<nw; iw++){
			out[ir][iw] = 0.;
		}
	}
}
