/*
	显示
*/
#include <petsc.h>
#include <math.h>
#define Pai  3.14159265358979

/*
      初始设定各种参数与类型
*/

int main(int argc, char **argv)
{

	PetscMPIInt		rank;
	PetscInt		i, j, size, iteration_num, Istart, Iend, col[3];
	PetscScalar		one = 1.0, dx = 0.01, dt = 0.00001, 
					ro = 1.0, c = 1.0, l = 1.0, k = 1.0, a, b,
					value[3], val, fval;
	Vec				u_n, u_n+1, f;
	Mat 			A;
	
	PetscCall(PetscInitialize(&argc,&argv,(char*)0,NULL));
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

/*
      输入dx, dt, l, ro, c,k
*/
	
	PetscOptionsGetScalar(NULL,NULL,"-dx",&dx,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-dt",&dt,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-l",&l,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-ro",&ro,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-c",&c,NULL);
                PetscOptionsGetScalar(NULL,NULL,"-k",&k,NULL);

/*
      定义向量与矩阵的各种系数
*/

	size 		  = 1/dx+1;
	iteration_num = 1/dt+1;
	a   	  = -k*dt/(ro*c*dx*dx);
	b 		  = -dt/(ro*c);

/*
      初始化向量
*/

	PetscCall(VecCreate(PETSC_COMM_WORLD,&u_n+1));
	PetscCall(VecSetSizes(u_n+1,PETSC_DECIDE,size));
	PetscCall(VecSetFromOptions(u_n+1));
	PetscCall(VecDuplicate(u_n+1,&u_n));
	PetscCall(VecDuplicate(u_n,&f));

/*
      设置u_n+1
*/	

	// PetscCall(VecGetOwnershipRange(u_n+1, &Istart, &Iend));
	for (i=1; i<size-1; i++)
                {
		val = exp(i*dx);
		PetscCall(VecSetValues(u_n+1,1,&i,&val,INSERT_VALUES));
	}
	PetscCall(VecAssemblyBegin(u_n+1));
                PetscCall(VecAssemblyEnd(u_n+1));
	// PetscCall(VecView(u_n+1,PETSC_VIEWER_STDOUT_WORLD));

/*
       设置f
*/

	// PetscCall(VecGetOwnershipRange(f, &Istart, &Iend));
	for (i=1; i<size-1; i++) 
                {
		val = sin(l*i*dx*Pai);
		PetscCall(VecSetValues(f,1,&i,&val,INSERT_VALUES));
	}
	PetscCall(VecAssemblyBegin(f));
                PetscCall(VecAssemblyEnd(f));
	// PetscCall(VecView(f,PETSC_VIEWER_STDOUT_WORLD));

/*
       初始化矩阵A
*/
	
	PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  	PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,size,size));
  	PetscCall(MatSetFromOptions(A));
  	PetscCall(MatMPIAIJSetPreallocation(A,3,PETSC_NULL,3,PETSC_NULL));
	PetscCall(MatSetUp(A));

	PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

	if (!Istart) 
                {
                Istart = 1;
                i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
                PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
                }
                    
                if (Iend == size)
                {
                Iend = size-1;
                i    = size-1; col[0] = size-2; col[1] = size-1; value[0] = -1.0; value[1] = 2.0;
                PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
                }

/*
        除了两行特判之外其他都符合一行三个元素 
*/    
         
                value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
                for (i=Istart; i<Iend; i++)
                {
                col[0] = i-1; col[1] = i; col[2] = i+1;
                PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
                }

	PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  	PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
	// PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
	
	
/*
        隐式计算 
*/  

                for(i=0;i<iteration_num;i++)
                {
		PetscCall(MatMult(A,u_n+1,u_n));
                                PetscCall(VecScale(u_n,a));
                                PetscCall(VecAXPY(u_n,b,f));
                                PetscCall(VecAXPY(u_n,1.0,u_n+1));
		PetscCall(VecSetValue(u_n,0,0.0,INSERT_VALUES));
		PetscCall(VecSetValue(u_n,size-1,0.0,INSERT_VALUES));
		PetscCall(VecAssemblyBegin(u_n));
		PetscCall(VecAssemblyEnd(u_n));

		PetscCall(VecCopy(u_n,u_n+1));
	}
	PetscCall(VecView(u_n,PETSC_VIEWER_STDOUT_WORLD));


	PetscPrintf(PETSC_COMM_WORLD,"size =  %D\n", size);

/*
        解构
*/
	
	PetscCall(VecDestroy(&u_n));
	PetscCall(VecDestroy(&u_n+1));
	PetscCall(VecDestroy(&f));
	PetscCall(MatDestroy(&A));
	PetscCall(PetscFinalize());
	return 0;
}