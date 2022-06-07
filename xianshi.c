/*
	显示
*/
#include <petsc.h>
#include <math.h>
#define Pai  3.14159265358979323846

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
	a   	  = k*dt/(ro*c*dx*dx);
	b 		  = dt/(ro*c);

/*
      初始化向量
*/

	PetscCall(VecCreate(PETSC_COMM_WORLD,&u_n));
	PetscCall(VecSetSizes(u_n,PETSC_DECIDE,size));
	PetscCall(VecSetFromOptions(u_n));
	PetscCall(VecDuplicate(u_n,&u_n+1));
	PetscCall(VecDuplicate(u_n+1,&f));

/*
      设置u_n
*/	

	// PetscCall(VecGetOwnershipRange(u_n, &Istart, &Iend));
	for (i=1; i<size-1; i++)
                {
		val = exp(i*dx);
		PetscCall(VecSetValues(u_n,1,&i,&val,INSERT_VALUES));
	}
	PetscCall(VecAssemblyBegin(u_n));
                PetscCall(VecAssemblyEnd(u_n));
	// PetscCall(VecView(u_n,PETSC_VIEWER_STDOUT_WORLD));

/*
       设置f
*/

	// PetscCall(VecGetOwnershipRange(f, &Istart, &Iend));
	for (i=1; i<size-1; i++) 
                {
		val = b * sin(l*i*dx*Pai);
		PetscCall(VecSetValues(f,1,&i,&val,INSERT_VALUES));
	}
	PetscCall(VecAssemblyBegin(f));
                PetscCall(VecAssemblyEnd(f));
	// PetscCall(VecView(f,PETSC_VIEWER_STDOUT_WORLD));


/*
        删除
*/
	
	PetscCall(VecDestroy(&u_n));
	PetscCall(VecDestroy(&u_n+1));
	PetscCall(VecDestroy(&f));
	PetscCall(MatDestroy(&A));
	PetscCall(PetscFinalize());
	return 0;
}
