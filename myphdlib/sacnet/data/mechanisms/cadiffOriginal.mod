

NEURON {
	SUFFIX cadiff
	USEION ca READ ica, cai WRITE cai
	GLOBAL cainf, taur, num
}

CONSTANT {
	FARADAY = 96489				
}

PARAMETER {
			
	cainf	= 0.0001	
}
ASSIGNED {
	ica
	taur 
	num
}
STATE {
	cai	 
}

INITIAL {
	cai = cainf

}

BREAKPOINT {

	SOLVE state METHOD euler
	
}
FUNCTION rate(ica1){
	if(ica1<-.00005){
		num=18
		taur=10
	}else{
		num=1
		taur=500
	}
}
DERIVATIVE state {
	rate(ica)
	cai'=-(100000)*num*ica/(2 * FARADAY )-(cai-cainf)/taur
	
}
