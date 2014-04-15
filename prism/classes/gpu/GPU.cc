
extern "C" {
#include <CL/opencl.h>
#include <stdio.h>
#include "aes/aes.c"
#include "shared/shared_consts.h"
#include "random_c/MersenneTwister.h"
#include "GPU.h"

JNIEXPORT jintArray JNICALL Java_gpu_GPU_initializeMersenneTwister
  (JNIEnv * env, jobject obj, jint num_prngs, jint seed )
{
	int i;
	//simulate __rnd__init arg "length"
	const int length = __MT_LENGTH;
	//simulate __rnd__init arg "id"
	const int id =  __MT_ID;
	const int array_length = num_prngs * (length + 1);
	cl_uint mother_state[__MOTHER_LENGTH];
	cl_uint *mem = (cl_uint*) calloc( array_length + __OFFSET + 3, sizeof(cl_uint) );
	__rnd_mother_init1( mother_state, seed );
	for( i = 0; i < array_length; i++ )
		mem[i + __OFFSET] = __rnd_mother( mother_state );
	for( i = 0; i < num_prngs; i++ )
	{
		aes_init( mem + i * (length + 1) + __OFFSET, (length + 3) & ~3, i + 1 );
		if( id < __PER_WORKITEM_IDS )
			mem[ (i + 1) * (length + 1) + __OFFSET - 1 ] = 0;
	}
	mem[0] = 0;
	mem[1] = id;
	mem[2] = num_prngs;
	/*
	prng->__device_state = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * (array_length + __OFFSET), mem, errcode_ret );
	prng->__host_state = NULL;
	prng->__num_prngs = num_prngs;
	prng->__info = __RND_DEVICE_ALLOC;
	prng->__context = context;
	prng->__ID = id;*/
	//we will change this to just copy array
	//make an assumption that sizeof(cl_uint)
	jintArray initValues = env->NewIntArray((array_length + __OFFSET) );  // allocate
   	if (NULL == initValues) return NULL;
	//cl_uint is just a 32-bit unsigned integer, so it should work...
	env->SetIntArrayRegion( initValues, 0 , (array_length + __OFFSET), (int*)mem);  // copy
	free(mem);
   	return initValues;
}

}
