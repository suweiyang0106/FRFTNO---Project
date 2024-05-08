# Fractional Fourier transform neural operator(FRFTNO) -Project
This project uses FRFT, a nonstationary random field, and several partial differential equations to prove it is better than FNO.<br />
Result:<br />
FRFTNO has 36%(1D)/21%(2D)/16%(3D) less l2 error than Fourier neural operator(FNO).:<br />
Documents:<br />
Old FNO: FNO vanilla <br />
Improved FNO: FNO_allmode <br />
FRFT: FRFTNO fixedarch <br />
Datasets: <br />
Burger/Poisson/Wave equations <br />
Warning: Burger equation dataset upwind scheme might be wrong due to negative value from the initial state.<br />
