# Command file to run ctfplotter
$ctfplotter -StandardInput
InputStack g.ali
AngleFile g.tlt
DefocusFile g.defocus
AxisAngle 85.3
PixelSize 0.25
ExpectedDefocus 4000
Voltage 300
SphericalAberration 2.7
