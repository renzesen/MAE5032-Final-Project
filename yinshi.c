#include <petscksp.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <math.h>
#include<petscerror.h>
#define Pai  3.14159265358979

/*
      初始设定各种参数与类型
*/

int main(int argc,char **args)
{
  PetscInt       i, col[3], rstart, rend, nlocal, rank, iter_num = 0;
  PetscInt       size, end, index;      
  PetscReal      ro = 1.0, c = 1.0, k = 1.0, l = 1.0;   
  PetscReal      a, b;   
  PetscReal      dx= 0.00001,dt = 0.00001, t = 0.0;
  PetscScalar    zero = 0.0, value[3], val, fval, data[3];
  Vec               u_now, u_next, f, tem;
  Mat               A;  
  KSP            ksp;  
  PC             pc; 
  PetscBool      restart = PETSC_FALSE; 
  PetscViewer   h5; 
  PetscErrorCode ierr;

/*
     初始化petsc与MPI
*/

  ierr =PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr =MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

/*
     读取参数
*/

     
   ierr =PetscOptionsGetReal(NULL,NULL,"-dx",&dx,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetReal(NULL,NULL,"-l",&l,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetReal(NULL,NULL,"-ro",&ro,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetReal(NULL,NULL,"-c",&c,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetReal(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
   ierr =PetscOptionsGetBool(NULL,NULL,"-restart",&restart,NULL);CHKERRQ(ierr);    
    

/*
      定义系数并输出
*/

  size                 = 1/dx+1;
  a   	        = k*dt/(ro*c*dx*dx);
  end                = size;
  b                    = dt/(ro*c);
  
   ierr =PetscPrintf(PETSC_COMM_WORLD,"a =  %g, b = %g\n", a, b);CHKERRQ(ierr);
   ierr =PetscPrintf(PETSC_COMM_WORLD,"dx =  %g\n", dx);CHKERRQ(ierr);
   ierr =PetscPrintf(PETSC_COMM_WORLD,"dt =  %g\n", dt);CHKERRQ(ierr);
   ierr =PetscPrintf(PETSC_COMM_WORLD,"restart =  %g\n", restart);CHKERRQ(ierr);
   ierr =PetscPrintf(PETSC_COMM_WORLD,"size =  %g\n", size);CHKERRQ(ierr);

/*
      初始化  Vec u_now, u_next, f
*/

   ierr =VecCreate(PETSC_COMM_WORLD,&u_now);CHKERRQ(ierr);
   ierr =VecSetSizes(u_now,PETSC_DECIDE,size);CHKERRQ(ierr);
   ierr =VecSetFromOptions(u_now);CHKERRQ(ierr);
   ierr =VecDuplicate(u_now,&u_next);CHKERRQ(ierr);
   ierr =VecDuplicate(u_next,&f);CHKERRQ(ierr);

   ierr =VecCreate(PETSC_COMM_WORLD,&tem);CHKERRQ(ierr);
   ierr =VecSetSizes(tem, 3, PETSC_DECIDE);CHKERRQ(ierr);
   ierr =VecSetFromOptions(tem);CHKERRQ(ierr);

   ierr =VecGetOwnershipRange(u_next, &rstart, &rend);CHKERRQ(ierr);
   ierr =VecGetLocalSize(u_next,&nlocal);CHKERRQ(ierr);

/*
       设置 Mat A
*/
	
   ierr =MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
   ierr =MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,size,size);CHKERRQ(ierr);
   ierr =MatSetFromOptions(A);CHKERRQ(ierr);
   ierr =MatMPIAIJSetPreallocation(A,3,PETSC_NULL,3,PETSC_NULL);CHKERRQ(ierr);
   ierr =MatSetUp(A);CHKERRQ(ierr);

   ierr =MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

  if (!rstart) 
  {
  rstart = 1;
  i      = 0; col[0] = 0; col[1] = 1; value[0] = 1+2.0*a; value[1] = -a;
   ierr =MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
                    
  if (rend == size)
  {
  rend = size-1;
  i    = size-1; col[0] = size-2; col[1] = size-1; value[0] = -a; value[1] = 1+2.0*a;
   ierr =MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  value[0] = -a; value[1] = 1+2.0*a; value[2] = -a;
  for (i=rstart; i<rend; i++)
  {
  col[0] = i-1; col[1] = i; col[2] = i+1;
   ierr =MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

   ierr =MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr =MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr =MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
/*
      判断是否需要重读还是新建Mat 
*/

  if(restart)
  {   
      ierr =PetscViewerHDF5Open(PETSC_COMM_WORLD,"explicit.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);   
      ierr =PetscObjectSetName((PetscObject) u_now, "explicit-vector");CHKERRQ(ierr); 
      ierr =PetscObjectSetName((PetscObject) tem, "explicit-necess-data");CHKERRQ(ierr);  
      ierr =VecLoad(tem, h5);CHKERRQ(ierr);    
      ierr =VecLoad(u_now, h5);CHKERRQ(ierr); 
      ierr =PetscViewerDestroy(&h5);CHKERRQ(ierr);   
      index=0;   
      ierr =VecGetValues(tem,1,&index,&dx);CHKERRQ(ierr);    
      index=index+1;    
      ierr =VecGetValues(tem,1,&index,&dt);CHKERRQ(ierr);    
      index=index+1;    
      ierr =VecGetValues(tem,1,&index,&t);CHKERRQ(ierr);   
      index= 0;   
  }
  else
  {    

/*
      设置  Vec u_now
*/	

     ierr =VecGetOwnershipRange(u_now, &rstart, &rend);CHKERRQ(ierr);
     ierr =VecGetLocalSize(u_now,&nlocal);CHKERRQ(ierr);
     ierr =VecSet(u_now,zero);CHKERRQ(ierr);
    if(rank == 0)
    {  
                for (i=1; i<size-1; i++)
                {
      		val = exp(i*dx);
      		 ierr =VecSetValues(u_now,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
     	}
    }
     ierr =VecAssemblyBegin(u_now);CHKERRQ(ierr);
     ierr =VecAssemblyEnd(u_now);CHKERRQ(ierr);
     ierr =VecView(u_now,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

/*
       设置  Vec f
*/
  
   ierr =VecGetOwnershipRange(f, &rstart, &rend);CHKERRQ(ierr);
   ierr =VecGetLocalSize(f,&nlocal);CHKERRQ(ierr);
   ierr =VecSet(f,zero);CHKERRQ(ierr);
  if(rank == 0)
  {  
	for (i=1; i<size-1; i++) 
                {
		fval = sin(l*i*dx*Pai);
		 ierr =VecSetValues(f,1,&i,&fval,INSERT_VALUES);CHKERRQ(ierr);
	}
  }
   ierr =VecAssemblyBegin(f);CHKERRQ(ierr);
   ierr =VecAssemblyEnd(f);CHKERRQ(ierr);
   ierr =VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

/*
        设置ksp 与 pc 
*/ 

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);   
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);   
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);    
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);    
  ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);   
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);    
	
/*
        隐式计算 
*/  

  while(PetscAbsReal(t)<=2.0)
  {    

     t += dt;   
      ierr = VecAXPY(u_now,1.0,f);CHKERRQ(ierr);    
      ierr = KSPSolve(ksp,u_now,u_next);CHKERRQ(ierr);    
      ierr =VecSetValue(u_next,0,0.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr =VecSetValue(u_next,size-1,0.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr =VecAssemblyBegin(u_next);CHKERRQ(ierr);
      ierr =VecAssemblyEnd(u_next);CHKERRQ(ierr);

      ierr =VecCopy(u_next,u_now);CHKERRQ(ierr);
  
      ierr =VecView(u_next,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  

     iter_num += 1;  
     if((iter_num % 10) == 0)
     {   
     
        data[0] = dx; data[1] = dt; data[2] = t;    
         ierr =VecSet(tem,zero);CHKERRQ(ierr);   
        for(index=0;index<3;index++)
        {    
        val = data[index];    
         ierr =VecSetValues(tem,1,&index,&val,INSERT_VALUES);CHKERRQ(ierr);   
        }
         ierr =VecAssemblyBegin(tem);CHKERRQ(ierr);    
         ierr =VecAssemblyEnd(tem);CHKERRQ(ierr);   

         ierr =PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);   
         ierr =PetscViewerHDF5Open(PETSC_COMM_WORLD,"explicit.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);    
         ierr =PetscObjectSetName((PetscObject) u_now, "explicit-vector");CHKERRQ(ierr);    
         ierr =PetscObjectSetName((PetscObject) tem, "explicit-necess-data");CHKERRQ(ierr);    
         ierr =VecView(tem, h5);CHKERRQ(ierr);   
         ierr =VecView(u_now, h5);CHKERRQ(ierr);   
         ierr =PetscViewerDestroy(&h5);CHKERRQ(ierr);    
     }  
  }

   ierr =VecView(u_now,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    

/*
        解构
*/
	
	 ierr =VecDestroy(&u_now);CHKERRQ(ierr);
	 ierr =VecDestroy(&u_next);CHKERRQ(ierr);
	 ierr =VecDestroy(&f);CHKERRQ(ierr);
                 ierr =VecDestroy(&tem);CHKERRQ(ierr);
	 ierr =MatDestroy(&A);CHKERRQ(ierr);
	
                 ierr =PetscFinalize();CHKERRQ(ierr);
                return ierr;
}	
	
     


     












