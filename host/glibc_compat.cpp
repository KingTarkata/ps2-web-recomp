// Binds the few libm functions whose symbol version glibc 2.43 bumped back to
// the baseline version, so a runner cross-compiled on a newer distribution
// starts on an older one (Switchroot's Ubuntu 24.04 ships glibc 2.39). Only
// built when the toolchain file asks for it.
extern "C" {
__asm__(".symver ps2_acosf_old,acosf@GLIBC_2.17");
__asm__(".symver ps2_asinf_old,asinf@GLIBC_2.17");
__asm__(".symver ps2_atan2f_old,atan2f@GLIBC_2.17");
__asm__(".symver ps2_coshf_old,coshf@GLIBC_2.17");
__asm__(".symver ps2_sinhf_old,sinhf@GLIBC_2.17");
float ps2_acosf_old(float);
float ps2_asinf_old(float);
float ps2_atan2f_old(float, float);
float ps2_coshf_old(float);
float ps2_sinhf_old(float);
float acosf(float x) { return ps2_acosf_old(x); }
float asinf(float x) { return ps2_asinf_old(x); }
float atan2f(float y, float x) { return ps2_atan2f_old(y, x); }
float coshf(float x) { return ps2_coshf_old(x); }
float sinhf(float x) { return ps2_sinhf_old(x); }
}
