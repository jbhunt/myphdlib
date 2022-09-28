
NEURON {
    
	SUFFIX kv
	USEION k READ ek WRITE ik
	RANGE n, h, ntau, htau,gk, gkbar,ninf, hinf
	 
	 
	GLOBAL n_50,n_slope,h_50,h_slope,ntau_50,ntau_slope,htau_50,htau_slope,ntau_max,htau_max
	GLOBAL v_shift,tau_shift,nN
}

PARAMETER {
	gkbar = 0.1   	(pS/um2)	: 0.12 mho/cm2
	ek=-80
	n_50=-50
	n_slope=0.1
	h_50=-50
	h_slope=0.2
	ntau_50=-50
	ntau_slope=0.1
	htau_50=-50
	htau_slope=0.1
	ntau_max=2
	htau_max=20	
	v 		(mV)
	v_shift=0
	tau_shift=1
	nN=1
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
	PI	= (pi) (1)
} 

ASSIGNED {
	ik 		(mA/cm2)
	gk		(pS/um2)
	ninf 		
	hinf	
	ntau:_inf
	htau:_inf
}
 

STATE { n h }:ntau htau}

INITIAL {
	rates(v+v_shift)
	n = ninf
	h = hinf
	:ntau=ntau_inf
	:htau=htau_inf
	
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gk = gkbar*(n)^nN*h
	ik = (1e-4) * gk * (v - ek)
} 




DERIVATIVE states {
    rates(v+v_shift)      
    n' =  (ninf-n)/(ntau)
    h' =  (hinf-h)/(htau)
}

PROCEDURE rates(vm) {  
	:Activation
	ninf = 1/(1+exp( (vm-n_50)*n_slope ) )
	ntau = ntau_max/(1+exp( (vm-ntau_50)*ntau_slope ) )
	
	:Deactivation
	hinf = 1/(1+exp( (vm-h_50)*h_slope ) )
	htau = htau_max/(1+exp( (vm-htau_50)*htau_slope ) )
}

