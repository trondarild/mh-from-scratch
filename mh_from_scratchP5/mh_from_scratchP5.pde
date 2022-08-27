//
//

float normal(float mean, float std){
  return randomGaussian()*std + mean;
}

float[] normal(float mean, float std, int size){
  float[] retval = zeros(size);
  for (int i = 0; i < size; ++i) {
    retval[i] = normal(mean, std);
  }
  return retval;
}

float[] transition_model(float[] x) {
  float[] retval = zeros(2);
  retval[0] = x[0];
  retval[1] = abs(normal(x[1], 0.5));
  return retval;
}

float prior(float[] x){
  if(x[1] <= 0) return 0.;
  return 1.;
}

boolean acceptance(float x, float x_new){
  if(x_new > x) return true;
  float accept = random(0, 1);
  return (accept < exp(x_new-x));
}

float log_lik_normal(float[] x, float[] data){
  float a = log(x[1] * sqrt(2*PI));
  float[] b = divide(pow(subtract(data, x[0]), 2), 2*sq(x[1]));
  float[] c = subtract(multiply(-1., b), a);
  return sumArray(c);
}

int num = 100;
float std = 75;

interface MHInterface{
  float log_likelihood(float[] x, float[] data);
  float[] transition_model(float[] x);
  boolean acceptance(float x, float x_new);
  float prior(float [] x);
}

void setup(){
  size(400, 400);
  frameRate(1);
}

void draw(){
  float[] xs = normal(0, std, num);
  float[] ys = normal(0, std, num);
  background(51);
  fill(200);
  pushMatrix();
  translate(width/2, height/2);
  for (int i = 0; i < num; ++i) {
    circle(xs[i], ys[i], 10);  
  }
  popMatrix();
  
  
}
