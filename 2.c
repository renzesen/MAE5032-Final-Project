static char help[] = "Inverse power iteration for a matrix";

#include <petscksp.h>
#include "petscmat.h"

/*
      初始设定各种参数与类型
  */

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscScalar     num = 0.0, value[3];
  Vec                z, y, x;
  Mat               A,B;
  PetscReal       y_norm = 0.0, y_norm_last, pluser, lam, err;
  MPI_Comm   comm;
  PetscInt         i, k, col[3], rstart, rend, n = 8, nlocal, iterator_num = 100;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-inum",&iterator_num,NULL));

  comm = MPI_COMM_WORLD;
  KSP ksp;
  KSPType type = KSPCG;
  KSPCreate(comm, &ksp); 
  KSPSetType(ksp, type); 
  KSPSetFromOptions(ksp);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));


/*
      创建向量
  */

  PetscCall(VecCreate(comm,&z));
  PetscCall( VecSetSizes(z,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(z));  
  PetscCall(VecCreate(comm,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecCreate(comm,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));


  PetscCall(VecGetOwnershipRange(z, &rstart, &rend));
  PetscCall(VecGetLocalSize(z,&nlocal));

/*
      初始化矩阵A B
  */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,nlocal,nlocal,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 4, PETSC_NULL, 4, PETSC_NULL));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,nlocal,nlocal,n,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatMPIAIJSetPreallocation(B, 4, PETSC_NULL, 4, PETSC_NULL));
  PetscCall(MatSetUp(B));

  PetscCall(VecSet(z, num));
  PetscCall(VecSetValue(z, 0, 1, INSERT_VALUES));

  PetscCall(VecAssemblyBegin(z));
  PetscCall(VecAssemblyEnd(z));



/*
      创建矩阵A
  */

    if (!rstart)
    {
        rstart = 1;
        i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
        PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }
    
    if (rend == n) 
   {
        rend = n-1;
        i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
        PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }

/*
      A赋值
  */

    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=rstart; i<rend; i++) 
   {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  KSPSetOperators(ksp, A, A);
  KSPSetTolerances(ksp, 1.0e-6, 1.0e-50, 1.0e5, 100);

/*
      运算
  */
    for(k = 0; k < iterator_num; k++)
  {
        KSPSolve(ksp, z, y);
        y_norm_last = y_norm;

        VecNorm(y, NORM_2, &y_norm);
        pluser = 1.0 / y_norm;
        VecAXPBY(z, pluser, 0, y);

        err = fabs(y_norm - y_norm_last);
        if(err < 1e-6)
        {
            break;
        } 
    }

  PetscCall(VecView(z,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

/* 
        求解
  */

    KSPSolve(ksp, z, y);
    MatMult(A, y, x);
    VecDot(x, y, &lam);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The running time is %.10lf\n", (double)(end-start)/CLOCKS_PER_SEC));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n", err, k));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Lambda =  %g\n", 1.0/lam));

/* 
        释放
  */
    PetscCall(VecDestroy(&z));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(PetscFinalize());
    return 0;
}