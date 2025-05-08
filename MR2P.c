#include <math.h>
#include <rsf.h>

// OPENMP support
#ifdef _OPENMP
#include <omp.h>
#endif

void rec2plane1p(sf_complex ***out, sf_complex ****in, float **vel, float px,
		 float ow , float dw , int nw ,
		 float orx, float drx, int nrx,
		 float ory, float dry, int nry,
		 float os , float ds , int ns );
void zeroout(sf_complex ***out, int nw, int nrx, int nry);

/*------------------------------------------------------------*/
int main(int argc, char* argv[])
{
        /* Variables */
	int np, nw, nrx, nry, ns, nx, nz;
    float op, dp, px;
    float ow, dw;
    float orx, drx;
	float ory, dry;
    float os, ds;

    float **vel;

    /* I/O files */
    sf_file Fin =NULL;
    sf_file Fvel=NULL;
    sf_file Fout=NULL;

    sf_axis aw, arx, ary, as, ap, ax, az;
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

	sf_warning("np: %d",np);
	sf_warning("dp: %g",dp);
	sf_warning("op: %g",op);
	
    /*---------------------------------------------------*/
    Fin  = sf_input ("in" ); sf_settype(Fin,SF_COMPLEX);
    Fvel = sf_input ("vel");
    Fout = sf_output("out"); sf_settype(Fout,SF_COMPLEX);

    aw  = sf_iaxa(Fin,1);
    arx = sf_iaxa(Fin,2);
	ary = sf_iaxa(Fin,3);
	as  = sf_iaxa(Fin,4);
	ax  = sf_iaxa(Fvel,2);
	az  = sf_iaxa(Fvel,1);
	ap  = sf_maxa(np,op,dp);

	nw  = sf_n(aw ); dw  = sf_d(aw ); ow  = sf_o(aw );
    nrx = sf_n(arx); drx = sf_d(arx); orx = sf_o(arx);
    nry = sf_n(ary); dry = sf_d(ary); ory = sf_o(ary);
	ns  = sf_n(as ); ds  = sf_d(as ); os  = sf_o(as );
	nx = sf_n(ax);
	nz = sf_n(az);

	sf_warning("ns: %d",ns);
	sf_warning("ds: %g",ds);
	sf_warning("os: %g",os);
	sf_warning("nrx: %d",nrx);
	sf_warning("nw: %d",nw);
	sf_warning("nry: %d",nry);

    sf_oaxa(Fout,aw ,1);
    sf_oaxa(Fout,arx,2);
	sf_oaxa(Fout,ary,3);
	sf_oaxa(Fout,ap ,4);

    sf_complex ****in;
	in  = sf_complexalloc4(nw,nrx,nry,ns);
	vel = sf_floatalloc2(nz,nx);
    sf_complex ***out;
	out = sf_complexalloc3(nw,nrx,nry);

    sf_complexread(in[0][0][0],nw*nrx*nry*ns,Fin);
    sf_floatread(vel[0],nz*nx,Fvel);

    /*---------------------------------------------------*/
    /*Main Loop*/

    /*All Shots*/
	/*Phase Encode*/
    for (int ip=0; ip<np; ip++){
        px = (float)(ip)*dp + op;

		sf_warning("IP of NP @ P: %d %d %f",ip,np,px);

		zeroout(out, nw, nrx, nry);

        rec2plane1p(out, in, vel, px,
                    ow , dw , nw ,
                    orx, drx, nrx,
                    ory, dry, nry,
                    os , ds , ns );

    	sf_complexwrite(out[0][0], nw*nrx*nry, Fout);
		
		//zeroout(out, nw, nrx, nry);
    }

	free(***in); free(**in); free(*in);  free(in);
	            free(**out); free(*out); free(out);
	exit(0);
}

void rec2plane1p(sf_complex ***out, sf_complex ****in, float **vel, float px,
		float ow , float dw , int nw ,
		float orx, float drx, int nrx,
		float ory, float dry, int nry,
		float os , float ds , int ns )
{		
	float xdt = ds*px;
	
	int is,iry,irx,iw,iv;
	float ww,tx,vx;
	
	//sf_warning("px: %g",px);

	tx=0.;

        for (is=0; is<ns; is++){
			//sf_warning("is: %d",is);
			//float sx = (is-1.)*ds+os;
			//tx = (float)(is)*xdt;
			tx += xdt*(vel[400][0]/vel[400][is]);
			//tx += xdt;
			//vx = (float) vel[400][is]/vel[400][0];
			//sf_warning("vx: %g",vx);
#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic,10)		\
    private(irx,iry,iw,ww)				\
    shared(xdt,tx,vx,out,in,dw,ow,nw,nrx,nry)
#endif
			for (iry=0; iry<nry; iry++){
				//float rx=(irx-1.)*drx+orx;
				//int ix = SF_MIN(SF_MAX((floor)((sx+rx-orx)/drx),0),nrx-1);
			
				for (irx=0; irx<nrx; irx++){
					//float ry=(iry-1.)*dry+ory;
					//int iy = SF_MIN(SF_MAX((floor)((sx+ry-ory)/dry),0),nry-1);
					for (iw=0; iw<nw; iw++){
						ww = (float)(iw)*dw+ow;
						//float tx = exp((float)(is)*px*ww);

						out[iry][irx][iw] += in[is][iry][irx][iw]*sf_cmplx(cosf(2.*SF_PI*ww*tx),sinf(2.*SF_PI*ww*tx));
				}
			}
		}
	}
}

void zeroout(sf_complex ***out, int nw, int nrx, int nry)
{

	int iry,irx,iw;

#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic)		\
    private(irx,iry,iw)				\
    shared(out,nw,nry,nrx)
#endif
	for (iry=0; iry<nry; iry++){ 
		for (irx=0; irx<nrx; irx++){
			for (iw=0; iw<nw; iw++){
				out[iry][irx][iw] = 0.;
			}
		}
	}
}
