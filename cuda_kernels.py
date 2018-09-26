import numpy as np
import cupy as cp

cu_eigh = cp.ElementwiseKernel(
    'float32 a, float32 b, float32 c, float32 d, float32 e, float32 f, float32 eig1, float32 eig2, float32 eig3, float32 vec1, float32 vec2, float32 vec3',
    '',
    '''
    float p1, q, p2, p, B00, B01, B02, B10, B11, B12, B20, B21, B22, r, phi, norm;
    p1 = pow(b,2) + pow(c,2) + pow(e,2);
    if (p1 == 0){
      eig1 = a;
      eig2 = d;
      eig3 = f;
    } else {
      q = (a + d + f) / 3;
      p2 = (a - q)*(a-q) + (d-q)*(d-q) + (f-q)*(f-q) + 2 * p1;

      p = sqrt(p2 / 6);
      B00 = (1 / p) * (a - q);
      B01 = (1 / p) * b;
      B02 = (1 / p) * c;
      B10 = B01;
      B11 = (1 / p) * (d - q);
      B12 = (1 / p) * e;
      B20 = B02;
      B21 = B12;
      B22 = (1 / p) * (f - q);
      r = (B00*(B11*B22 - B12*B21) - B01*(B10*B22 - B20*B12) + B02*(B10*B21 - B11*B20)) / 2;

      if (r <= -1){
        phi = M_PI / 3;
      } else if (r >= 1){
        phi = 0;
      } else {
        phi = acos(r) / 3;
      }

      eig1 = q + 2 * p * cos(phi);
      eig3 = q + 2 * p * cos(phi + (2 * M_PI / 3));
      eig2 = 3 * q - eig1 - eig3;
    }

    vec1 = (a - eig1) * (a - eig2) + b*b + c*c;
    vec2 = b*(a - eig2) + (d-eig1)*b + e*c;
    vec3 = c*(a-eig2) + e*b + (f - eig1)*c;

    norm = sqrt(vec1*vec1 + vec2*vec2 + vec3*vec3);
    if (norm == 0)
    {
      vec1 = 1;
      vec2 = 0;
      vec3 = 0;
    } 
    else
    {
      vec1 /= norm;
      vec2 /= norm;
      vec3 /= norm;
    }


    ''',
    'cu_eigh')
