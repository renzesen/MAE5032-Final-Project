/*
	显示
*/
#include <petsc.h>
#include <math.h>


/*
      初始设定各种参数与类型
*/

int main(int argc, char **argv)
{

	PetscMPIInt		rank;
	PetscInt		                i, j, size, iteration_num；
	PetscScalar	                one = 1.0, dx = 0.01, dt = 0.00001, 
			                p = 1.0, c = 1.0, l = 1.0, k = 1.0；
	Vec				u_last, u_now, f;
	Mat 			A;
	
	PetscCall(PetscInitialize(&argc,&argv,(char*)0,NULL));
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

/*
      输入dx, dt, l, p, c, k
*/
	
	PetscOptionsGetScalar(NULL,NULL,"-dx",&dx,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-dt",&dt,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-l",&l,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-p",&p,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-c",&c,NULL);
                PetscOptionsGetScalar(NULL,NULL,"-k",&k,NULL);
/*
      定义向量与矩阵的各种系数
*/

	size 		  = 1/dx;
	iteration_num = 1/dt;
	lambda   	  = k*dt/(p*c*dx*dx);
	gamma 		  = dt/(p*c);

/*
      初始化向量
*/

	PetscCall(VecCreate(PETSC_COMM_WORLD,&u_last));
	PetscCall(VecSetSizes(u_last,PETSC_DECIDE,size));
	PetscCall(VecSetFromOptions(u_last));
	PetscCall(VecDuplicate(u_last,&u_now));
	PetscCall(VecDuplicate(u_now,&f));


/*
        拆解
*/
	
	PetscCall(VecDestroy(&u_last));
	PetscCall(VecDestroy(&u_now));
	PetscCall(VecDestroy(&f));
	PetscCall(MatDestroy(&A));
	PetscCall(PetscFinalize());
	return 0;
}
