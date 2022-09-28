COMMENT
an synaptic current with alpha function conductance defined by
        i = g * (v - e)      i(nanoamps), g(microsiemens);
        where
         g = 0 for t < onset and
         g = gmax * (t - onset)/tau * exp(-(t - onset - tau)/tau)
          for t > onset
this has the property that the maximum value is gmax and occurs at
 t = delay + tau.
ENDCOMMENT
					       
NEURON {
	POINT_PROCESS AlphaSynapseG
	RANGE onset, tau, gmax, e, i, g
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	onset=0 (ms)
	tau=.1 (ms)	<1e-3,1e6>
	gmax=0 	(uS)	<0,1e9>
	e=0	(mV)
	g=0 (uS)
}

ASSIGNED { i (nA) }

BREAKPOINT {
	
	 i =(1e-3)* g*(v - e)
}

