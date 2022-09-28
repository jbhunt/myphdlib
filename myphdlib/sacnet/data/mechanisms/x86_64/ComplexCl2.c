/* Created by Language version: 7.5.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ComplexCl2
#define _nrn_initial _nrn_initial__ComplexCl2
#define nrn_cur _nrn_cur__ComplexCl2
#define _nrn_current _nrn_current__ComplexCl2
#define nrn_jacob _nrn_jacob__ComplexCl2
#define nrn_state _nrn_state__ComplexCl2
#define _net_receive _net_receive__ComplexCl2 
#define state state__ComplexCl2 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define tau1 _p[0]
#define tau2 _p[1]
#define e _p[2]
#define thres _p[3]
#define amp _p[4]
#define scaling _p[5]
#define i _p[6]
#define g _p[7]
#define ves (_p + 8)
#define thyme (_p + 13)
#define numreleased _p[18]
#define rando _p[19]
#define after _p[20]
#define before _p[21]
#define test _p[22]
#define alpha _p[23]
#define probrelease _p[24]
#define A _p[25]
#define B _p[26]
#define t1 _p[27]
#define DA _p[28]
#define DB _p[29]
#define _g _p[30]
#define _nd_area  *_ppvar[0]._pval
#define capre	*_ppvar[2]._pval
#define _p_capre	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_gettest();
 static double _hoc_getnumreleased();
 static double _hoc_initthyme();
 static double _hoc_initves();
 static double _hoc_sumves();
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "gettest", _hoc_gettest,
 "getnumreleased", _hoc_getnumreleased,
 "initthyme", _hoc_initthyme,
 "initves", _hoc_initves,
 "sumves", _hoc_sumves,
 0, 0
};
#define gettest gettest_ComplexCl2
#define getnumreleased getnumreleased_ComplexCl2
#define initthyme initthyme_ComplexCl2
#define initves initves_ComplexCl2
#define sumves sumves_ComplexCl2
 extern double gettest( );
 extern double getnumreleased( double , double );
 extern double initthyme( double );
 extern double initves( double );
 extern double sumves( );
 /* declare global and static user variables */
#define ca_baseline ca_baseline_ComplexCl2
 double ca_baseline = 0.0001;
#define numves numves_ComplexCl2
 double numves = 5;
#define regentime regentime_ComplexCl2
 double regentime = 900;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "numves_ComplexCl2", "integer",
 "i", "nanoamp",
 0,0
};
 static double A0 = 0;
 static double B0 = 0;
 static double delta_t = 0.01;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "ca_baseline_ComplexCl2", &ca_baseline_ComplexCl2,
 "numves_ComplexCl2", &numves_ComplexCl2,
 "regentime_ComplexCl2", &regentime_ComplexCl2,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.5.0",
"ComplexCl2",
 "tau1",
 "tau2",
 "e",
 "thres",
 "amp",
 "scaling",
 0,
 "i",
 "g",
 "ves[5]",
 "thyme[5]",
 "numreleased",
 "rando",
 "after",
 "before",
 "test",
 "alpha",
 "probrelease",
 0,
 "A",
 "B",
 0,
 "capre",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 31, _prop);
 	/*initialize range parameters*/
 	tau1 = 0;
 	tau2 = 0;
 	e = -65;
 	thres = 0;
 	amp = 0.09;
 	scaling = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 31;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ComplexCl2_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
  hoc_register_prop_size(_mechtype, 31, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ComplexCl2 /home/jbhunt/Dropbox/Code/Python/lib/sacnet/sacnet/data/mechanisms/x86_64/ComplexCl2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static double *_temp1;
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
double initves (  double _lnum ) {
   double _linitves;
 double _ll ;
 _ll = 0.0 ;
   while ( _ll < numves ) {
     ves [ ((int) _ll ) ] = _lnum ;
     _ll = _ll + 1.0 ;
     }
   
return _linitves;
 }
 
static double _hoc_initves(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  initves (  *getarg(1) );
 return(_r);
}
 
double initthyme (  double _lnum ) {
   double _linitthyme;
 double _ll ;
 _ll = 0.0 ;
   while ( _ll < numves ) {
     thyme [ ((int) _ll ) ] = _lnum ;
     _ll = _ll + 1.0 ;
     }
   
return _linitthyme;
 }
 
static double _hoc_initthyme(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  initthyme (  *getarg(1) );
 return(_r);
}
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   DA = - A / tau1 ;
   DB = - B / tau2 ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 DA = DA  / (1. - dt*( ( - 1.0 ) / tau1 )) ;
 DB = DB  / (1. - dt*( ( - 1.0 ) / tau2 )) ;
  return 0;
}
 /*END CVODE*/
 
static int state () {_reset=0;
 {
   DA = - A / tau1 ;
   DB = - B / tau2 ;
   }
 return _reset;}
 
double gettest (  ) {
   double _lgettest;
 _lgettest = .5 ;
   
return _lgettest;
 }
 
static double _hoc_gettest(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  gettest (  );
 return(_r);
}
 
double getnumreleased (  double _lcapre1 , double _lt2 ) {
   double _lgetnumreleased;
 double _lw , _ll ;
 if ( _lcapre1 > thres ) {
     _ll = 0.0 ;
     while ( _ll < numves ) {
       if ( ves [ ((int) _ll ) ]  == 0.0 ) {
         rando = scop_random ( ) ;
         if ( rando <= ( _lt2 - thyme [ ((int) _ll ) ] ) / regentime ) {
           ves [ ((int) _ll ) ] = 1.0 ;
           }
         }
       _ll = _ll + 1.0 ;
       }
     if ( _lcapre1 > thres ) {
       probrelease = ( _lcapre1 - thres ) * tan ( .349 ) ;
       }
     else {
       probrelease = 0.0 ;
       }
     before = sumves ( _threadargs_ ) ;
     _lw = 0.0 ;
     while ( _lw < 5.0 ) {
       if ( ves [ ((int) _lw ) ]  == 1.0 ) {
         rando = scop_random ( ) ;
         if ( rando <= probrelease ) {
           ves [ ((int) _lw ) ] = 0.0 ;
           thyme [ ((int) _lw ) ] = _lt2 ;
           }
         }
       _lw = _lw + 1.0 ;
       }
     after = sumves ( _threadargs_ ) ;
     numreleased = before - after ;
     }
   else {
     numreleased = 0.0 ;
     }
   
return _lgetnumreleased;
 }
 
static double _hoc_getnumreleased(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  getnumreleased (  *getarg(1) , *getarg(2) );
 return(_r);
}
 
double sumves (  ) {
   double _lsumves;
 double _li ;
 _li = 0.0 ;
   _lsumves = 0.0 ;
   while ( _li < numves ) {
     _lsumves = _lsumves + ves [ ((int) _li ) ] ;
     _li = _li + 1.0 ;
     }
   
return _lsumves;
 }
 
static double _hoc_sumves(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r =  sumves (  );
 return(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  A = A0;
  B = B0;
 {
   capre = ca_baseline ;
   A = 0.0 ;
   B = 0.0 ;
   t1 = 0.0 ;
   initves ( _threadargscomma_ 1.0 ) ;
   initthyme ( _threadargscomma_ - 1.0 ) ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   if ( t > t1 ) {
     getnumreleased ( _threadargscomma_ capre , t ) ;
     A = A + numreleased * amp ;
     B = B + numreleased * amp ;
     t1 = t1 + 1.0 ;
     }
   g = A - B ;
   i = ( 1e-3 ) * g * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 { error =  euler(_ninits, 2, _slist1, _dlist1, _p, &t, dt, state, &_temp1);
 if(error){fprintf(stderr,"at line 83 in file ComplexCl2.mod:\n    \n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 2; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(A) - _p;  _dlist1[0] = &(DA) - _p;
 _slist1[1] = &(B) - _p;  _dlist1[1] = &(DB) - _p;
_first = 0;
}
