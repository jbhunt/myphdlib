: Graded Synapse with first order binding kinetics

NEURON {
POINT_PROCESS SimpleCl

    RANGE e,g

    POINTER capre

NONSPECIFIC_CURRENT i
}

PARAMETER {
	e=-65
    capre=0
    ca_baseline=0.0001
    l=20
 
}

ASSIGNED {
	v (millivolt)
    i (nanoamp)
	g
}
 
BREAKPOINT {
	g=capre
	i = (1e-3)*g * (v - e)
}
 
INITIAL {
	g  =0
    capre=ca_baseline
}

