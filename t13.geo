// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 13
//
//  Remeshing an STL file without an underlying CAD model
//
// -----------------------------------------------------------------------------

// Let's merge an STL mesh that we would like to remesh.
Merge "t13_data.stl";

// We first classify ("color") the surfaces by splitting the original surface
// along sharp geometrical features. This will create new discrete surfaces,
// curves and points.

DefineConstant[
  // Angle between two triangles above which an edge is considered as sharp
  angle = {40, Min 20, Max 120, Step 1,
    Name "Parameters/Angle for surface detection"},
  // For complex geometries, patches can be too complex, too elongated or too
  // large to be parametrized; setting the following option will force the
  // creation of patches that are amenable to reparametrization:
  forceParametrizablePatches = {0, Choices{0,1},
    Name "Parameters/Create surfaces guaranteed to be parametrizable"},
  // For open surfaces include the boundary edges in the classification process:
  includeBoundary = 1,
  // Force curves to be split on given angle:
  curveAngle = 180
];
ClassifySurfaces{angle * Pi/180, includeBoundary, forceParametrizablePatches,
                 curveAngle * Pi / 180};

// Create a geometry for all the discrete curves and surfaces in the mesh, by
// computing a parametrization for each one
CreateGeometry;

// In batch mode the two steps above can be performed with `gmsh t13.stl
// -reparam 40', which will save `t13.msh' containing the parametrizations, and
// which can thus subsequently be remeshed.

// Create a volume as usual
Surface Loop(1) = Surface{:};
Volume(1) = {1};

// We specify element sizes imposed by a size field, just because we can :-)
funny = DefineNumber[0, Choices{0,1},
  Name "Parameters/Apply funny mesh size field?" ];

Field[1] = MathEval;
If(funny)
  Field[1].F = "2*Sin((x+y)/5) + 3";
Else
  Field[1].F = "4";
EndIf
Background Field = 1;
