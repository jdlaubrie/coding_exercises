module parameters
  implicit none
  real(kind=8),parameter :: zero=0.d0,one=1.d0,two=2.d0,three=3.d0,four=4.d0
  real(kind=8),parameter :: pi=3.141592653589d0
!geometric parameters
  real(kind=8),parameter :: radius=0.023d0,thickness=0.002d0
!numeric parameters
  integer,parameter :: tstep=100,maxits=1000
  real(kind=8),parameter :: step=0.01d0,tol=500.d0
end module parameters
!********************************************************
program tube
  use parameters
  implicit none
  integer :: inc,counter
  real(kind=8) :: tot_press,pressure,time
  real(kind=8) :: stress_circ0,stress_circ1,stress_circ2
  real(kind=8) :: stress_long0,stress_long1,stress_long2
  real(kind=8) :: stress_n0,stress_n1,stress_n2
  real(kind=8) :: stress_theta0,stress_theta1,stress_theta2
  real(kind=8) :: stress_z0,stress_z1,stress_z2
  real(kind=8) :: stress_r0,stress_r1,stress_r2
  real(kind=8) :: lambda_theta0,lambda_theta1,lambda_theta2,lambda_r
  real(kind=8) :: derivative
!open files for out
  open(unit=10,file='analitic.dat',status='replace')
!total pressure
  tot_press = 10000.d0
!loading history
  do inc=0,tstep   !number of steps
    time = real(inc)/real(tstep)
    pressure = zero + tot_press*time
    !initial roots
    lambda_theta1 = one
    lambda_theta2 = one + step
    counter = 0
    stress: do !stress cycle
      !calculate stress in the wall tube
      call walltube(pressure,lambda_theta1,stress_n1,stress_circ1,stress_long1)
      call walltube(pressure,lambda_theta2,stress_n2,stress_circ2,stress_long2)
      !calculate stress from anisotropic model
      call stresses(time,pressure,lambda_theta1,stress_r1,stress_theta1,stress_z1)
      call stresses(time,pressure,lambda_theta2,stress_r2,stress_theta2,stress_z2)
      !jacobian matrix
      derivative=(stress_theta2-stress_circ2-stress_theta1+stress_circ1) &
& /(lambda_theta2-lambda_theta1)
      !looking for new roots
      lambda_theta0 = lambda_theta1 - (stress_theta1-stress_circ1)/derivative
      call walltube(pressure,lambda_theta0,stress_n0,stress_circ0,stress_long0)
      call stresses(time,pressure,lambda_theta0,stress_r0,stress_theta0,stress_z0)
      !checking convergence
      if (abs(stress_theta0-stress_circ0)<tol) exit stress
      !updating roots
      lambda_theta2 = lambda_theta1
      lambda_theta1 = lambda_theta0
      !limit for de number of iterations
      counter = counter + 1
      if (counter>maxits) then
        write(*,*) 'error: many iterations'
        exit stress
      end if
    end do stress !stress cycle
!writing
    lambda_r = one/(lambda_theta0)
    write(10,*) inc,lambda_r,lambda_theta0,stress_r0,stress_theta0,stress_z0
  end do !number of steps

  close(10)
  call system ('gnuplot plotting.plt')
end program tube
!****************subroutine stresses*********************************************
subroutine stresses(time,pressure,lambda_theta,stress_r,stress_theta,stress_z)
  use parameters
  implicit none
  !exteral variables
  real(kind=8),intent(in) :: lambda_theta,time,pressure
  real(kind=8),intent(out) :: stress_z,stress_theta,stress_r
  !internal variables
  integer :: i
  real(kind=8) :: stress_theta_e,stress_theta_c,stress_theta_ci,stress_theta_m
  real(kind=8) :: stress_z_e,stress_z_c,stress_z_ci,stress_z_m
  real(kind=8) :: lambda_r,lambda_c,lambda_m
  real(kind=8) :: alpha
  real(kind=8) :: g_r,g_theta,g_z,g_c,g_m
  real(kind=8) :: c10,k1c,k2c,k1m,k2m
  real(kind=8) :: fiber_c(4),fiber_m
  real(kind=8) :: den_e,den_c(4),den_m,den_tot,rho_e,rho_c(4),rho_m
!parameter or constant for modeling hyperelastic behavior and deposition tensor
  c10 = 300.0e3
  k1c = 100.0e3
  k2c = 0.01
  k1m = 10.0e3
  k2m = 0.01
  g_z = one + time*0.1d0
  g_theta = one + time*0.1832064
  g_r = one + time*(one/(g_theta*g_z) - one)
  g_c = one + time*0.1d0
  g_m = one + time*0.1d0
! densities
  den_e = 241.5d0
  den_c(1) = 65.1d0
  den_c(2) = 65.1d0
  den_c(3) = 260.4d0
  den_c(4) = 260.4d0
  den_m = 157.5d0
  den_tot = den_e + den_c(1) + den_c(2) + den_c(3) + den_c(4) + den_m
  rho_e = den_e/den_tot
  rho_c(1) = den_c(1)/den_tot
  rho_c(2) = den_c(2)/den_tot
  rho_c(3) = den_c(3)/den_tot
  rho_c(4) = den_c(4)/den_tot
  rho_m = den_m/den_tot
!fiber directon for every fiber family
  fiber_c(1) = zero
  fiber_c(2) = pi/two
  fiber_c(3) = pi/four
  fiber_c(4) = -pi/four
  fiber_m = pi/two
!incompresibility
  lambda_r = one/(lambda_theta)
!*******************stress in R direction***********************
!stress in circumferential direction
  stress_r = -pressure
!*******************stress in THETA direction***********************
  !elstin
  stress_theta_e = rho_e*two*c10*(g_theta**two*lambda_theta**two-g_r**two**lambda_r**two)
  !collagen
  stress_theta_c = zero
  do i=1,4
   alpha = fiber_c(i)
   lambda_c = g_c*sqrt(lambda_theta**two*sin(alpha)**two+cos(alpha)**two)  
   stress_theta_ci = rho_c(i)*k1c*lambda_c**two*(lambda_c**two-one) &
& *exp(k2c*(lambda_c**two-one)**two)*sin(alpha)**two
   stress_theta_c = stress_theta_c + stress_theta_ci
  end do
  !smc
  alpha = fiber_m
  lambda_m = g_m*sqrt(lambda_theta**two*sin(alpha)**two+cos(alpha)**two)
  stress_theta_m = rho_m*k1m*lambda_m**two*(lambda_m**two-one) &
& *exp(k2m*(lambda_m**two-one)**two)*sin(alpha)**two
!stress in circumferential direction
  stress_theta = stress_theta_e + stress_theta_c + stress_theta_m - pressure
!******************stress in Z direction************************
  !elastin
  stress_z_e = rho_e*two*c10*(g_z**two-g_r**two*lambda_r**two)
  !collagen
  stress_z_c = zero
  do i=1,4
   alpha = fiber_c(i)
   lambda_c = g_c*sqrt(lambda_theta**two*sin(alpha)**two+cos(alpha)**two)  
   stress_z_ci = rho_c(i)*k1c*lambda_c**two*(lambda_c**two-one) &
& *exp(k2c*(lambda_c**two-one)**two)*cos(alpha)**two
   stress_z_c = stress_z_c + stress_z_ci
  end do
  !smc
  alpha = fiber_m
  lambda_m = g_m*sqrt(lambda_theta**two*sin(alpha)**two+cos(alpha)**two)
  stress_z_m = rho_m*k1m*lambda_m**two*(lambda_m**two-one) &
& *exp(k2m*(lambda_m**two-one)**two)*cos(alpha)**two
!stress in longitudinal direction
  stress_z = stress_z_e + stress_z_c + stress_z_m - pressure
!************************************************************
end subroutine stresses
!**********subroutine walltube********************************************
subroutine walltube(pressure,lambda_theta,stress_r,stress_circ,stress_long)
  use parameters
  implicit none
  real(kind=8),intent(in) :: pressure,lambda_theta
  real(kind=8),intent(out) :: stress_r,stress_circ,stress_long
  real(kind=8) :: ri,t
  real(kind=8) :: lambda_t,lambda_r
!geometry
  lambda_t = lambda_theta
  lambda_r = one/lambda_theta
  ri=lambda_t*radius
  t =lambda_r*thickness
!stresses in tube
  stress_r = -pressure
  stress_circ = pressure*ri/t
  stress_long = pressure*ri/(two*t)

end subroutine walltube
