
: L-type Ca current for Retinal Ganglion Cell from Benison et al (2001)
: M.Migliore Nov. 2001
: modelDB accession 3457

NEURON {
	SUFFIX calrgc
	USEION ca READ eca WRITE ica
	RANGE  gbar, ica, shift, test, minf, mtau, a, b, m
	:GLOBAL minf, mtau
	
}

PARAMETER {
	gbar = .01   	(mho/cm2)	
	shift= 20	
			
	eca	=100	(mV)            : must be explicitly def. in hoc
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
    v           (mV)
	ica 		(mA/cm2)
	minf		mtau (ms)
	test
	a 
	b 

}
 

STATE { m }

BREAKPOINT {
        SOLVE states METHOD cnexp
	test=v-eca
	ica = gbar*m*m*((v  - eca))
} 

INITIAL {
	trates(v)
	m=minf  
}

DERIVATIVE states {   
        trates(v)        :possible shift
        m' = (minf-m)/mtau
}

PROCEDURE trates(vm) {  
        

	:a = trap0(vm,2,0.061,12.5)
	:b = 0.058*exp(-(vm-10)/4)
	a = trap0(vm+shift,3,0.061,12.5)
	b = 0.058*exp(-(vm+shift-10)/14)
	minf = a/(a+b)
	mtau = 1/(a+b)
	
	

}

FUNCTION trap0(vm,th,a,q) {
	if (fabs(vm-th) > 1e-6) {
	        trap0 = a * (vm - th) / (1 - exp(-(vm - th)/q))
	} else {
	        trap0 = a * q
 	}
}	