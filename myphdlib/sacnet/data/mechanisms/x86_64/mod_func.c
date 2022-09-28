#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _AlphaSynapseG_reg(void);
extern void _Bipg_reg(void);
extern void _Bip_reg(void);
extern void _cadiffOriginal_reg(void);
extern void _Calrgc_reg(void);
extern void _ComplexCl2g_reg(void);
extern void _ComplexCl2_reg(void);
extern void _kv_reg(void);
extern void _readica_reg(void);
extern void _SimpleCl_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," AlphaSynapseG.mod");
    fprintf(stderr," Bipg.mod");
    fprintf(stderr," Bip.mod");
    fprintf(stderr," cadiffOriginal.mod");
    fprintf(stderr," Calrgc.mod");
    fprintf(stderr," ComplexCl2g.mod");
    fprintf(stderr," ComplexCl2.mod");
    fprintf(stderr," kv.mod");
    fprintf(stderr," readica.mod");
    fprintf(stderr," SimpleCl.mod");
    fprintf(stderr, "\n");
  }
  _AlphaSynapseG_reg();
  _Bipg_reg();
  _Bip_reg();
  _cadiffOriginal_reg();
  _Calrgc_reg();
  _ComplexCl2g_reg();
  _ComplexCl2_reg();
  _kv_reg();
  _readica_reg();
  _SimpleCl_reg();
}
