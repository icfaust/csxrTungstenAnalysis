#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/*************INITIALIZE PYTHON MODULE****************/

// functions
static PyObject *gauss(PyObject *self, PyObject *args);
static PyObject *gauss2(PyObject *self, PyObject *args);
static PyObject *gaussjac(PyObject *self, PyObject *args);
static PyObject *gaussPoissonMLE(PyObject *self, PyObject *args);


//module constuctor
static struct PyMethodDef meth[] = {
  {"gauss", gauss, METH_VARARGS, "generates gaussians"},
  {"gauss2", gauss2, METH_VARARGS, "generates gaussians using integral, and not maximum value (multiply maximum height by standard deviation and square root of pi)"},
  {"gaussjac", gaussjac, METH_VARARGS, "generates numpy array of derivative of gaussians with minimization of RMSE"},
  {"gpmle", gaussPoissonMLE, METH_VARARGS, "fit gaussian profiles using maximum likelihood estimation assuming Poisson data errors"},
  {NULL, NULL, 0, NULL} /* not entirely sure what this line does, but is necessary */
};

PyMODINIT_FUNC init_gauss (void){
   (void)Py_InitModule("_gauss", meth);
   import_array(); // the all important numpy call
}

/*****************************************************/

static PyObject *gauss(PyObject *self, PyObject *args){
  int fail = 0, nd, i, j, lenidx, lenxdata, size;
  PyObject *arg1 = NULL, *xdata = NULL, *arg2 = NULL, *inp = NULL; //input Python objects
  PyObject *output;
  PyArray_Descr *descr = NULL;
  npy_intp *dims;
  double *xdptr, *inptr, *outptr;
  
  // Check inputs
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;

  // CONDITION INPUTS TO DOUBLES
  xdata = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xdata == NULL) fail = 1;
  if (!fail) inp = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (inp == NULL) fail = 1;

  // CONTSTRUCT OUTPUT
  if (xdata != NULL){
    nd = PyArray_NDIM(xdata);
    dims = PyArray_DIMS(xdata);
    descr = PyArray_DESCR(xdata);
    output = PyArray_Zeros(nd, dims, descr, 0);
    if(output == NULL) fail = 1;
  }

  // main body of operation, 
  if (!fail){
    //initialize final looping variables
    size = PyArray_Size(inp);    
    lenidx = size;

    //initialize final looping variables
    size = PyArray_Size(xdata);    
    lenxdata = size;
    
    
    xdptr = (double*)PyArray_DATA(xdata);
    inptr = (double*)PyArray_DATA(inp);
    outptr = (double*)PyArray_DATA(output);

    //Now everything is prepared for the C function, everything else has been python/numpy c-api work
    
    for(i=0; i<lenidx; i=i+3){
      for(j=0; j<lenxdata; j++){
	outptr[j] = outptr[j] + inptr[i]*exp(-1*pow((xdptr[j]-inptr[i+1])/inptr[i+2],2));
      }
    }
    
    //output is a 1d function which is successively added to by gaussians (using math.h exponential)
    // I have yet how to decide how to force it to the correct datatype, but that will require some effort

  }


  /*Py_CLEAR(xdata);
    Py_CLEAR(inp);*/
  
  if (fail){

    return NULL;
  }else{

    PyArray_ENABLEFLAGS((PyArrayObject*)output, NPY_ARRAY_OWNDATA);
    return output;
  }
}


static PyObject *gauss2(PyObject *self, PyObject *args){
  int fail = 0, nd, i, j, lenidx, lenxdata, size;
  PyObject *arg1 = NULL, *xdata = NULL, *arg2 = NULL, *inp = NULL; //input Python objects
  PyObject *output;
  PyArray_Descr *descr = NULL;
  npy_intp *dims;
  double *xdptr, *inptr, *outptr;
  
  // Check inputs
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;

  // CONDITION INPUTS TO DOUBLES
  xdata = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xdata == NULL) fail = 1;
  if (!fail) inp = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (inp == NULL) fail = 1;

  // CONTSTRUCT OUTPUT
  if (xdata != NULL){
    nd = PyArray_NDIM(xdata);
    dims = PyArray_DIMS(xdata);
    descr = PyArray_DESCR(xdata);
    output = PyArray_Zeros(nd, dims, descr, 0);
    if(output == NULL) fail = 1;
  }

  // main body of operation, 
  if (!fail){
    //initialize final looping variables
    size = PyArray_Size(inp);    
    lenidx = size;

    //initialize final looping variables
    size = PyArray_Size(xdata);    
    lenxdata = size;
    
    
    xdptr = (double*)PyArray_DATA(xdata);
    inptr = (double*)PyArray_DATA(inp);
    outptr = (double*)PyArray_DATA(output);

    //Now everything is prepared for the C function, everything else has been python/numpy c-api work
    
    for(i=0; i<lenidx; i=i+3){
      for(j=0; j<lenxdata; j++){
	outptr[j] = outptr[j] + inptr[i]*exp(-1*pow((xdptr[j]-inptr[i+1])/inptr[i+2],2))/inptr[i+2]/1.7724538;
      }
    }
    
    //output is a 1d function which is successively added to by gaussians (using math.h exponential)
    // I have yet how to decide how to force it to the correct datatype, but that will require some effort

  }


  /*Py_CLEAR(xdata);
    Py_CLEAR(inp);*/
  
  if (fail){

    return NULL;
  }else{

    PyArray_ENABLEFLAGS((PyArrayObject*)output, NPY_ARRAY_OWNDATA);
    return output;
  }
}


static PyObject *gaussjac(PyObject *self, PyObject *args){
  int fail = 0, nd = 0, i=0, j, lenidx = 1, lenxdata= 1, tempidx;
  PyObject *arg1 = NULL, *xdata = NULL, *arg2 = NULL, *ydata = NULL, *arg3 = NULL, *inp = NULL; //input Python objects
  PyObject *output;
  PyArray_Descr *descr = NULL;
  npy_intp *dims;
  double *xdptr, *ydptr, *inptr, *outptr, temp, exponent;

  // Check inputs
  if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3)) return NULL;

  // CONDITION INPUTS TO DOUBLES
  xdata = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xdata == NULL) fail = 1;
  if (!fail) ydata = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (ydata == NULL) fail = 1;
  if (!fail) inp = PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if (inp == NULL) fail = 1;

  // CONTSTRUCT OUTPUT
  if (inp != NULL){
    nd = PyArray_NDIM(inp); // number of dimensions of xdata
    dims = PyArray_DIMS(inp); //pointer to the dimensions of xdata
  }
  
  npy_intp *dims2 = (npy_intp *)malloc(nd*sizeof(npy_intp));

  if (inp != NULL){
    for(i=0;i<nd;i++){
      dims2[i] = dims[i];
    }    
    descr = PyArray_DESCR(inp); 
    // need to expand this to a larger array which has the individual
    // add one value to the the dims first index, assumes that the index is flat.
    dims2[0] = dims2[0] + 1; 
    
    output = PyArray_Zeros(nd, dims2, descr, 0);
    if(output == NULL) fail = 1;
  }
    
  // main body of operation, 
  if (!fail){
    //initialize final looping variables
    tempidx = PyArray_Size(inp);    
    lenidx = tempidx;

    //initialize final looping variables
    tempidx = PyArray_Size(xdata);    
    lenxdata = tempidx;

    xdptr = (double*)PyArray_DATA(xdata);
    ydptr = (double*)PyArray_DATA(ydata);
    inptr = (double*)PyArray_DATA(inp);
    outptr = (double*)PyArray_DATA(output);
  }

  //initialize large c data array for vairous calculations
  
  double *arr = (double *)malloc((lenidx + 1)*(lenxdata)*sizeof(double)); // first index is the sum

  //Now everything is prepared for the C function, everything else has been python/numpy c-api work

    // step 1, store all individual contributions and total in two arrays
  if(!fail){
    // essentially a data copy with the inclusion of the offset
    for(j=0; j <lenxdata; j++){
      arr[j] =  ydptr[j]-inptr[0];
    }

    // step 2 calculate the point by point subtraction and initial derivative values
    
    for(i=1; i<lenidx; i=i+3){
      for(j=0; j<lenxdata; j++){
	exponent = (xdptr[j]-inptr[i+1])/inptr[i+2];
	temp = exp(-1*pow(exponent,2));
	arr[j] = arr[j] - temp*inptr[i];
	arr[j + lenxdata*(i)] = temp; // derivative of the magnitude
	arr[j + lenxdata*(i+1)] = 2*temp*inptr[i]*exponent/inptr[i+2]; // derivative of the offset
	arr[j + lenxdata*(i+2)] = 2*temp*inptr[i]*pow(exponent,2)/inptr[i+2]; // derivative of the width of the gaussian
	
      }
    }


    // generate the RMS value of the total function
    for(j=0; j<lenxdata; j++){
      *outptr = *outptr + pow(arr[j],2); // derivative of the magnitude
    }

    *outptr = pow(*outptr,.5); // do the square root for the RMS value

    // generate jacobian for baseline
    for(j=0; j<lenxdata; j++){
      outptr[1] = outptr[1] + arr[j]; // derivative of the magnitude
      // hannah loves ian very much
    }  // THIS IS THE NEW BASELINE CALCULATOR
    //printf(" %f %f \n",outptr[1]);
    outptr[1] = -1*outptr[1]/ *outptr;
    //printf(" %f %f \n",outptr[1],outptr[0]);
    
    // step 3 calculate the total RMS value of the data using step 2 data
    for(i=2; i<=lenidx; i++){
      tempidx = lenxdata*(i-1); // probably not a great idea to reuse variables this way, but as long as I note it, whatever
      for(j=0; j<lenxdata; j++){
	outptr[i] = outptr[i] + arr[j] * arr[j + tempidx]; // derivative of the magnitude
	// hannah loves ian very much
      }
      outptr[i] = -1* outptr[i]/ *outptr;  // do I need to apply a negative here??? currently looks like the chisquared
    }

    
    // step 4 calculate derivatives using the step 1, step 2 and step 3 data, part 1.

    // sum to get the 
  
  
    // step 4 calculate sum the various contributions

    // step 5 condition data for return
    
    //output is a 1d function which is successively added to by gaussians (using math.h exponential)
    // I have yet how to decide how to force it to the correct datatype, but that will require some effort

  }



  /*Py_CLEAR(xdata);
  Py_CLEAR(ydata);
  Py_CLEAR(inp);*/
  /*Py_XDECREF(xdata);
  Py_XDECREF(ydata);
  Py_XDECREF(inp);*/
  free(arr);
  free(dims2);
  
  if (fail){

    return NULL;
  }else{

    PyArray_ENABLEFLAGS((PyArrayObject*)output, NPY_ARRAY_OWNDATA);
    return output;
  }
}


static PyObject *gaussPoissonMLE(PyObject *self, PyObject *args){
  int fail = 0, nd = 0, i=0, j, lenidx = 1, lenxdata= 1, tempidx;
  PyObject *arg1 = NULL, *xdata = NULL, *arg2 = NULL, *ydata = NULL, *arg3 = NULL, *inp = NULL; //input Python objects
  PyObject *output;
  PyArray_Descr *descr = NULL;
  npy_intp *dims;
  double *xdptr, *ydptr, *inptr, *outptr, temp, exponent;

  // Check inputs
  if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3)) return NULL;

  // CONDITION INPUTS TO DOUBLES
  xdata = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xdata == NULL) fail = 1;
  if (!fail) ydata = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (ydata == NULL) fail = 1;
  if (!fail) inp = PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if (inp == NULL) fail = 1;

  // CONTSTRUCT OUTPUT
  if (inp != NULL){
    nd = PyArray_NDIM(inp); // number of dimensions of xdata
    dims = PyArray_DIMS(inp); //pointer to the dimensions of xdata
  }
  
  npy_intp *dims2 = (npy_intp *)malloc(nd*sizeof(npy_intp));

  if (inp != NULL){
    for(i=0;i<nd;i++){
      dims2[i] = dims[i];
    }    
    descr = PyArray_DESCR(inp); 
    // need to expand this to a larger array which has the individual
    // add one value to the the dims first index, assumes that the index is flat.
    // this will be the output of the value being minimized, and the gradients
    // with respect to each of the input parameters, hence + 1 (first value
    // is the minimization value)
    dims2[0] = dims2[0] + 1; 
    
    output = PyArray_Zeros(nd, dims2, descr, 0); //create output python array
    if(output == NULL) fail = 1;
  }
    
  // main body of operation, 
  if (!fail){
    //initialize final looping variables
    lenidx = PyArray_Size(inp);    

    //initialize final looping variables
    lenxdata = PyArray_Size(xdata);

    xdptr = (double*)PyArray_DATA(xdata);
    ydptr = (double*)PyArray_DATA(ydata);
    inptr = (double*)PyArray_DATA(inp);
    outptr = (double*)PyArray_DATA(output);
  }

  //initialize large c data array for various calculations
  
  double *arr = (double *)malloc((lenidx + 1)*(lenxdata)*sizeof(double)); // first index is the sum

  //Now everything is prepared for the C function, everything else has been python/numpy c-api work

  if(!fail){
    // essentially a data copy of the offset for the comparison array
    for(j=0; j <lenxdata; j++){
      arr[j] =  inptr[0];
    }

    // step 1 begin assembling the model data and the derivatives for every input
    
    for(i=1; i<lenidx; i=i+3){
      for(j=0; j<lenxdata; j++){
	exponent = (xdptr[j]-inptr[i+1])/inptr[i+2];
	temp = exp(-1*pow(exponent,2))/inptr[i+2]/1.7724538; // square root of pi hardcoded
	arr[j] = arr[j] + temp*inptr[i];
	arr[j + lenxdata*(i)] = temp; // derivative of the area under the curve (redefined from other programs above)
	arr[j + lenxdata*(i+1)] = 2*temp*inptr[i]*exponent/inptr[i+2]; // derivative of the offset
	arr[j + lenxdata*(i+2)] = temp*inptr[i]*(2*pow(exponent,2) - 1)/inptr[i+2]; // derivative of the width of the gaussian
	
      }
    }

    // step 2 generate the MLE for a Poisson distribution using completed model data
    *outptr = 0;
    for(j=0; j<lenxdata; j++){
      *outptr = *outptr + arr[j] - ydptr[j]*log(arr[j]);
    }
    //printf(" %e \n",*outptr);
    // step 3 generate jacobians using completed model data and jacobians
    
    // step 3 for baseline
    for(j=0; j<lenxdata; j++){
      outptr[1] = outptr[1] + (1 - ydptr[j]/arr[j]); // derivative of the magnitude
      // hannah loves ian very much
    }

    //printf(" %f %f \n",outptr[1],outptr[0]);
    
    // step 3 for gaussians
    for(i=2; i<=lenidx; i++){
      tempidx = lenxdata*(i-1);
      outptr[i] = 0;
      for(j=0; j<lenxdata; j++){
	outptr[i] = outptr[i] + (1 - ydptr[j]/arr[j])*arr[j + tempidx];
	// hannah loves ian very much
      }
    }
  }

  free(arr);
  free(dims2);
  
  if (fail){

    return NULL;
  }else{

    PyArray_ENABLEFLAGS((PyArrayObject*)output, NPY_ARRAY_OWNDATA);
    return output;
  }
}
