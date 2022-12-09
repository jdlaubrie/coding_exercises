using LinearAlgebra
module MyModule
   mutable struct equilibrium
      pressure::Real
      g_t::Real
      g_z::Real
      g_m::Real
      g_c::Real
   end
end
###############
# Calculate stresses in a thin-walled tube
###############
function ThinWalled(parameters,stretch)
   pressure=parameters.pressure
   R=10.0
   H=1.41
   #geometry
   stretch_t = stretch
   stretch_r = 1.0/stretch
   r=R
   h=H
   #stresses in tube
   stress_r = -pressure/2.0
   stress_circ = pressure*r/h
   stress_long = pressure*r/(2.0*h)
   return stress_r,stress_circ,stress_long
end
###############
# Calculate stresses with anisotropic model
###############
function WallMixture(parameters,stretch,multiplier)
# parameter or constant for modeling hyperelastic behavior and deposition tensor
   c10 = 72.0
   k1c = 1136.0
   k2c = 11.2
   k1m = 15.2
   k2m = 11.4
# densities
   den_c = zeros(Float64,4)
   den_e = 241.5e-9
   den_c[1] = 65.1e-9
   den_c[2] = 260.4e-9
   den_c[3] = 260.4e-9
   den_c[4] = 65.1e-9
   den_m = 157.5e-9
   den_tot = den_e + den_c[1] + den_c[2] + den_c[3] + den_c[4] + den_m
# fiber directon for every fiber family
   fiber_c = zeros(Float64,4)
   fiber_c[1] = pi/2.0
   fiber_c[2] = pi/4.0
   fiber_c[3] = -pi/4.0
   fiber_m = pi/2.0
#*******************About deformation gradients***********************
# deformation gradient tensor
   #Identity=Matrix{Float64}(I,3,3)
   F = Matrix{Float64}(I,3,3)
   F[1,1] = 1.0/stretch
   F[2,2] = stretch
   F[3,3] = 1.0
   Jaco=det(F)
# Elastin
   Gh_e = Matrix{Float64}(I,3,3)
   Gh_e[3,3] = parameters.g_z
   Gh_e[2,2] = parameters.g_t
   Gh_e[1,1] = 1.0/(Gh_e[2,2]*Gh_e[3,3])
   #total deposition stretch
   F_e = Matrix{Float64}(I,3,3)
   mul!(F_e,F,Gh_e)
   #right cauchy-green tensor for elastin
   b_e = Matrix{Float64}(I,3,3)
   mul!(b_e,F_e,transpose(F_e))
   #stresses for elastin
   stress_e = den_e*c10*b_e/Jaco
#Collagen
   stress_c = zeros(Float64,3,3)
   g_c = parameters.g_c
   for i in 1:4
      stress_ci = zeros(Float64,3,3)
      N0 = zeros(Float64,3)
      FN = zeros(Float64,3)
      N0 = [0.0,sin(fiber_c[i]),cos(fiber_c[i])]
      mul!(FN,F,N0)
      FN = g_c*FN
      innerFN = dot(FN,FN)
      mul!(stress_ci,FN,transpose(FN))
      stress_ci = den_c[i]*k1c*(innerFN-1.0)*exp(k2c*(innerFN-1.0)^2)*stress_ci/Jaco
      stress_c = stress_c + stress_ci
   end
#SMC
   s_act = 54.0e-3
   lambda_m = 1.4
   lambda_a = 1.0
   lambda_0 = 0.8
   stress_m = zeros(Float64,3,3)
   FN = zeros(Float64,3)
   g_m = parameters.g_m
   N0 = [0.0,sin(fiber_m),cos(fiber_m)]
   mul!(FN,F,N0)
   FN = g_m*FN
   innerFN = dot(FN,FN)
   mul!(stress_m,FN,transpose(FN))
   stress_m = den_m*(k1m*(innerFN-1.0)*exp(k2m*(innerFN-1.0)^2)+(s_act/(den_tot*innerFN))*(1.0-((lambda_m-lambda_a)/(lambda_m-lambda_0))^2))*stress_m/Jaco
# *******************stress in R direction***********************
   stress_r = stress_e[1,1] + multiplier
# *******************stress in THETA direction***********************
   stress_theta = stress_e[2,2] + stress_c[2,2] + stress_m[2,2] + multiplier
# ******************stress in Z direction************************
   stress_z = stress_e[3,3] + stress_c[3,3] + stress_m[3,3] + multiplier
# ************************************************************
   return stress_r,stress_theta,stress_z
end
###############
# Compute circumferential residual function
###############
function ResidualFunction(parameters,Variable)
   #call function to compute the stresses
   ThinRadiStress,ThinCircStress,ThinLongStress = ThinWalled(parameters,Variable[1])
   MixRadiStress,MixCircStress,MixLongStress = WallMixture(parameters,Variable[1],Variable[2])
   #compute the residual
   Residual = zeros(Float64,2)
   Residual[1] = MixCircStress-ThinCircStress
   Residual[2] = MixRadiStress-ThinRadiStress
   return Residual
end
###############
# Compute the derivative of a function
###############
function NumericalDerivative(parameters,Variable)
   hh = 1.0e-6
   DResidual = zeros(Float64,2,2)
   for j in 1:2
      forwardVar = zeros(Float64,2)
      backwardVar = zeros(Float64,2)
      forwardVar = [Variable[1],Variable[2]]
      backwardVar = [Variable[1],Variable[2]]
      for i in 1:2
         # Ridder's derivative method
         NTAB = 10
         BIG = 1.0e30
         CON = 1.4
         CON2 = CON*CON
         SAFE = 2.0
         A = zeros(Float64,NTAB,NTAB)
         forwardVar[j] = Variable[j]*(1.0+hh)
         backwardVar[j] = Variable[j]*(1.0-hh)
         A[1,1] = (ResidualFunction(parameters,forwardVar)[i]-ResidualFunction(parameters,backwardVar)[i])/(2.0*Variable[j]*hh)
         err = BIG
         for k in 2:NTAB
            hh=hh/CON
            forwardVar[j] = Variable[j]*(1.0+hh)
            backwardVar[j] = Variable[j]*(1.0-hh)
            A[1,k] = (ResidualFunction(parameters,forwardVar)[i]-ResidualFunction(parameters,backwardVar)[i])/(2.0*Variable[j]*hh)
            fac = CON2
            for l in 2:k
               A[l,k]=(A[l-1,k]*fac-A[l-1,k-1])/(fac-1.0)
               fac=CON2*fac
               errt=max(abs(A[l,k]-A[l-1,k]),abs(A[l,k]-A[l-1,k-1]))
               if errt<=err
                  err=errt
                  DResidual[i,j] = A[l,k]
               end
            end
            #if abs(A[k,k]-A[k-1,k-1])>=SAFE*err
            #   println("Failed at looking for derivatives with Ridders Method")
            #   break
            #end
         end
      end
   end
   return DResidual
end
###############
# Newton-Raphson method
###############
import .MyModule.equilibrium
parameters = equilibrium(13.332e-3,1.34,1.25,1.1,1.062)
number_of_increments=30
# initial guess
error=1.0
Variable = [1.0,-0.0001]
for Inc in 1:number_of_increments
global Variable
#global residual_norm
println("Increment $Inc")
factor=Inc/number_of_increments
parameters.pressure=13.332e-3*factor
parameters.g_t=1.0+0.34*factor
parameters.g_z=1.0+0.25*factor
parameters.g_m=1.0+0.10*factor
parameters.g_c=1.0+0.062*factor
Residual = ResidualFunction(parameters,Variable)
residual_norm = norm(Residual)
Iter=1
while residual_norm>1.0e-2
   Tangent = zeros(Float64,2,2)
   Delta_Sol = zeros(Float64,2)

   Tangent = NumericalDerivative(parameters,Variable)
   Tangent_inv = inv(Tangent)

   mul!(Delta_Sol,-Tangent_inv,Residual)
   Variable += Delta_Sol

   Residual = ResidualFunction(parameters,Variable)
   residual_norm = norm(Residual)
   error = abs(Delta_Sol[1]/Variable[1])

   Iter+=1
   if residual_norm>1.0e10 || Iter>50
      println("NEWTON-RAPHSON: The method did not converge")
      break
   end
end
   println("$Variable, $residual_norm, $Iter")
   if residual_norm>1.0e10 || Iter>50
      println("INCREMENTS: The method did not converge")
      break
   end
end
