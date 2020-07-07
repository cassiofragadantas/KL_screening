/*
 *  Usage: [W H objKL timeKL] = ccd_KL(V, k, max_iter, Winit, Hinit, trace);
 *
 * Given the nonnegative input matrix V, this code solves the following KL-NMF problem to find the low-rank approximation WH for V. 
 *
 *  min_{W>=0,H>=0} sum_{i,j} V_{ij}*log(V_{ij}/(WH)_{ij})
 *
 *  Input arguments
 *  	V: n by m nonnegative input matrix.
 *  	k: rank of output matrices W and H. 
 *  	max_iter: maximum iteration. 
 *  	Winit: k by n initial matrix for W. 
 *  	Hinit: k by m initial matrix for H. 
 *  	trace: 1: compute objective value per iteration. 
 *  		   0: do not compute objective value per iteration. (default)
 *
 *  Output arguments
 *  	W: k by n dense matrix.
 *  	H: k by m dense matrix.
 *  	objKL: objective values.
 *  	timeKL: time taken by this algorithm. 
 *
 */

#include "math.h"
#include "mex.h" 
#include <time.h>
#include <blas.h>

double obj(int n, int m, double *V, double *WH)
{
	double total = 0;
	for ( int i=0 ; i<n*m ; i++ )
		total = total + V[i]*log((V[i]+1e-5)/(WH[i]+1e-5))-V[i]+WH[i];
	return (total);
}

void update(int m, int k, double *Wt, double *WHt, double *Vt, double *H, double l1reg, double eps, double eps_y)
{
	int maxinner = 1;
    mwSignedIndex one = 1;
    mwSignedIndex m2 = m;
    
	for ( int q=0 ; q<k ; q++ )
	{
		for (int inneriter =0 ; inneriter<maxinner ; inneriter++)
		{
			double g=l1reg, h=0, tmp, s, oldW, newW, diff;

            // Calculate first and second derivatives for Newton's update
            for (int j=0; j<m ; j++ )
			{	
				tmp = (Vt[j]+eps_y)/(WHt[j]+eps);
				g = g + H[q*m + j]*(1-tmp); // 1-V/WH
				h = h + H[q*m + j]*H[q*m + j]*tmp/(WHt[j]+eps);    //V/WH^2
			}
            //couldn't avoid this for loop... even using BLAS dsbmv and ddot

			s = -g/h;
			oldW = Wt[q];
			newW = Wt[q]+s;
            
			//if ( newW < 1e-15)
			//	newW = 1e-15;
            //Testing: seems to work!
            if ( newW < 0)
				newW = 0;
            
			diff = newW-oldW;
			Wt[q] = newW;
            
            //Updating WHt (both ways work)
            //Option 1 - for loop
			//for ( int j=0 ; j<m ; j++)
			//	WHt[j] = WHt[j]+diff*H[q*m+j];
            //Option 2 - BLAS (faster)
            daxpy(&m2,&diff,H + q*m, &one, WHt, &one);
			if ( fabs(diff) < fabs(oldW)*0.5 )
				break;
		}
	}
}

void usage()
{
	printf("Error calling KL_l1_CoD_update.\n");
	//printf("Usage: [W H objKL timeKL] = ccd_KL(V, k, max_iter, Winit, Hinit, trace=0)\n");
    printf("Usage: [H] = KL_l1_CoD_update(V, W, Hinit, WH, l1reg=0)\n");

    // Model: V = W.'*H
    // Dimensions: V (n x m), W (k x n), H (k x m), in the regression case we have m=1
    //old usage: [W H objKL timeKL] = ccd_KL(V, k, max_iter, Winit, Hinit, trace=0)
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int i;
	double *V, *W, *H, *WH;
    double l1reg = 0, eps = 1e-15, eps_y = 1e-15;

	int n,m, k;
	double *outH, *outWH;

	// Check input/output number of arguments
	if ( nlhs > 2 || nrhs < 4 )
	{
		usage();
		printf("Number of input or output arguments are not correct.\n");
		return;
	}

    // Get input variables
	V = mxGetPr(prhs[0]);
    n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);
    
    W = mxGetPr(prhs[1]);
    k = mxGetN(prhs[1]);
    //W is (n x k), so the model is V = W*H
    if (mxGetM(prhs[1])!=n ) {
		usage();
		printf("Error: W should be a %d by %d matrix. \n", n, k);
		return;
	}
    
	H = mxGetPr(prhs[2]);
	if ( mxGetM(prhs[2]) != k || mxGetN(prhs[2])!=m ) {
		usage();
		printf("Error: H should be a %d by %d matrix. \n", k, m);
		return;
	}
    
    WH = mxGetPr(prhs[3]);
	if ( mxGetM(prhs[3]) != n || mxGetN(prhs[3])!=m ) {
		usage();
		printf("Error: WH should be a %d by %d matrix. \n", n, m);
		return;
	}

    if ( nrhs>4 )
        l1reg =  mxGetScalar(prhs[4]);

    if ( nrhs>5 )
        eps =  mxGetScalar(prhs[5]);
    
    if ( nrhs>6 )
        eps_y =  mxGetScalar(prhs[6]);
   
    // Update H
    if (m == 1){
        update(n,k,H,WH,V,W,l1reg,eps,eps_y);
    }else{ //case m>1
        for ( int i=0 ; i<m ; i++ )
        {
            double *Ht = &(H[i*k]); 
            double *wht = &(WH[i*n]);
            double *vt = &(V[i*n]);

            update(n,k,Ht,wht,vt,W,l1reg,eps,eps_y);

        }
    }

    /*Not necessary to generate output vectors, since the input are directly modified
    // Output H
	plhs[0] = mxCreateDoubleMatrix(k,m,mxREAL);
	outH=mxGetPr(plhs[0]);
	for ( i=0 ; i<k*m ; i++ )
		outH[i] = H[i];

    //Output WH
    plhs[1] = mxCreateDoubleMatrix(n,m,mxREAL);
    //plhs[1] = mxDuplicateArray(prhs[3]);
	outWH = mxGetPr(plhs[1]);
    for ( i=0 ; i<m*n ; i++ )
        outWH[i] = WH[i];
    //Why not? plhs[1] = mxDuplicateArray(WH);
    */
	return;
}
