: A gabaergic chloride channel that goes from presynaptic calcium concentration to quantized vesicles to channel conductance

NEURON{
    POINT_PROCESS ComplexCl2
    NONSPECIFIC_CURRENT i
    RANGE e,g,numreleased, ves, thyme, after, test,alpha, thres, i, probrelease, scaling, amp, tau1, tau2, rando, before
    POINTER capre
}

PARAMETER{
    tau1
    tau2
    e=-65
    capre=0
    ca_baseline=.0001
    thres=.0000          :1.4715
    amp=.09
    numves= 5 (integer)
    regentime=900: 300
    scaling=1:.000005
    
}
ASSIGNED{

	v (millivolt)
    i (nanoamp)
	g
    t1
    ves[5]
    thyme[5]
    numreleased
    rando
    after
    before
    test
    alpha
    probrelease
}
STATE{
    A 
    B
}
INITIAL{
    capre=ca_baseline
    A=0
    B=0
    t1=0
    :initvec(ves[],1)
    initves(1)
    initthyme(-1)

}
        COMMENT
        FUNCTION initvec(vec[],num){
            LOCAL l
            l=0
            while(l<numves){
                vec[l]=num
                l=l+1
            }
        }
        ENDCOMMENT

        FUNCTION initves(num) {
            LOCAL l
            l=0
            while(l<numves){
                ves[l]=num
                l=l+1
            }
        }
        FUNCTION initthyme(num) {
            LOCAL l
            l=0
            while(l<numves){
                thyme[l]=num

                l=l+1
            }
        }
BREAKPOINT{
    SOLVE state METHOD euler
    
    if (t>t1){ 
        getnumreleased(capre,t)
        A=A+numreleased*amp
        B=B+numreleased*amp
        t1=t1+1
    }

        :test=gettest()
    
    
        g=A-B
        i = (1e-3)*g * (v - e)  
    
}   
DERIVATIVE state {
	A'=-A/tau1
 	B'=-B/tau2 
}

        COMMENT
        FUNCTION getnumreleased(capre1, t1)(){
            if(capre1>thres){
                numreleased=1
            }else{
                numreleased=0
            }
        }
        ENDCOMMENT

FUNCTION gettest(){
    gettest= .5
    :gettest=scop_random()
    
}
FUNCTION getnumreleased(capre1,t2){
    LOCAL w, l
    if(capre1>thres){
        l=0
        while(l<numves){
            if (ves[l]==0){
                rando=scop_random()
                if(rando<=(t2-thyme[l])/regentime){
                    ves[l]=1
                }
            }
            l=l+1
        }
    
        if(capre1>thres){
            
            :probrelease=(capre1*capre1*capre1)/scaling
            probrelease= (capre1-thres)*tan(.349) : previously tan(1.52)
        }else{
            probrelease=0
        }
        
        :probrelease=1
        before=sumves()
        w=0
        while(w<5){
            if(ves[w]==1){
                rando=scop_random()
                if(rando<=probrelease){             :inequality?
                    ves[w]=0
                    thyme[w]=t2
                }
            }
            w=w+1
        }
        after=sumves()
        
        
        numreleased=before-after
    }else{
        numreleased=0
    }
}

FUNCTION sumves(){
    LOCAL i
    i=0
    sumves=0
    while(i<numves){
        sumves=sumves+ves[i]
        i=i+1
    }
}











   
