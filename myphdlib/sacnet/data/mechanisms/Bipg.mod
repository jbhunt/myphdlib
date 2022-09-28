: Graded Synapse with first order binding kinetics

NEURON {
POINT_PROCESS Bipg
	
	RANGE g1max,g2max,tau,e,g
	RANGE del,dur
	RANGE locx,locy,preX,preY

NONSPECIFIC_CURRENT i
}

PARAMETER {
	g1max=1
	g2max=0.2
	tau=20
	e=0
	locx=0				:location x
	locy=0				:location y
	preX=0
	preY=0				:link to presynaptic cell
	del=100
	dur=1000
}

ASSIGNED {
	v (millivolt)
	i (nanoamp)
	g
}
 
BREAKPOINT {

	if (t>del){
		g = g1max * (t - del)/tau * exp(-(t - del - tau)/tau)
	}
	if ((t>del)&&(t<del+dur)){
		g=g+g2max
	}
	i = (1e-3)*g * (v - e)
}
 
INITIAL {
	g  =0
}
 
